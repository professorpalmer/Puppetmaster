"""Optional CodeGraph integration.

CodeGraph (https://github.com/colbymchenry/codegraph) builds a local SQLite
index of a repository's symbols, references, and routes. When it's installed
and the target workspace has a `.codegraph/` directory, Puppetmaster workers
can query it to seed prompts with shared code intelligence instead of having
each worker rediscover the repo with grep/read passes.

This module is fully optional. Every helper returns gracefully when the
`codegraph` CLI is missing, the workspace is not initialized, or the query
times out, so adapters can call it without conditional plumbing.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.fs_permissions import mkdir_private, open_private


CODEGRAPH_COMMAND = "codegraph"
CODEGRAPH_PACKAGE = "@colbymchenry/codegraph"
MAX_CONTEXT_CHARS = 4000
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 30
DEFAULT_STATUS_TIMEOUT_SECONDS = 10
DEFAULT_QUERY_TIMEOUT_SECONDS = 15
DEFAULT_AFFECTED_TIMEOUT_SECONDS = 30
DEFAULT_FILES_TIMEOUT_SECONDS = 15
# Synchronous init (without --index) is the only path that should ever block
# the caller. The CodeGraph `init` step is fast (creates `.codegraph/` and a
# minimal scaffold); indexing is what takes minutes-to-hours. Indexing now
# dispatches through `puppetmaster_codegraph_index` as a background
# subprocess, so this timeout only bounds the small init step.
DEFAULT_INIT_TIMEOUT_SECONDS = 60


CODEGRAPH_MISSING_HINT = (
    "CodeGraph is unavailable because Node.js is not installed on this machine. "
    "Puppetmaster auto-provisions the CodeGraph CLI via `npx @colbymchenry/codegraph` "
    "whenever Node is present (no manual install needed), but it cannot run a Node CLI "
    "without a Node runtime. Install Node.js 18+ (https://nodejs.org) and retry; "
    "or, if you keep a global install, `npm install -g @colbymchenry/codegraph`. "
    "Set PUPPETMASTER_CODEGRAPH_NO_NPX=1 to disable the npx fallback."
)
CODEGRAPH_NOT_INITIALIZED_HINT = (
    "workspace is not initialized for CodeGraph. Run `puppetmaster_codegraph_init` "
    "or `codegraph init` in the target repository first."
)
CODEGRAPH_NATIVE_SQLITE_HINT = (
    "CodeGraph's native SQLite (better-sqlite3) is broken on this machine — almost "
    "always a Node ABI mismatch. The common Cursor trap: your terminal's Node "
    "(e.g. Homebrew v23, ABI 131) and Cursor's bundled Node (v22, ABI 127) "
    "have different ABIs, so a plain `npm rebuild` from your shell builds "
    "better-sqlite3 for the wrong runtime and Puppetmaster's MCP keeps falling "
    "back to slow WASM SQLite (you'll see `database is locked` errors). "
    "Easiest fix: run `python -m puppetmaster repair-codegraph` (or call the "
    "`puppetmaster_repair_codegraph` MCP tool) — it rebuilds better-sqlite3 "
    "against Cursor's Node automatically and verifies Backend: native. Manual "
    "equivalent: `CURSOR_NODE=\"/Applications/Cursor.app/Contents/Resources/app/"
    "resources/helpers/node\" && cd \"$(npm root -g)/@colbymchenry/codegraph\" && "
    "PATH=\"$(dirname \"$CURSOR_NODE\"):$PATH\" npm rebuild better-sqlite3`. "
    "Restart the Puppetmaster MCP server in Cursor afterwards."
)


def _npx_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_CODEGRAPH_NO_NPX")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _npx_codegraph_invocation() -> Optional[list[str]]:
    """Return ``[npx, -y, @colbymchenry/codegraph]`` when npx is available.

    This is the universal fallback that makes CodeGraph "always available" on
    any machine with Node, with zero manual setup: ``npx`` fetches the package
    into its cache on first use and reuses it thereafter. Because the package's
    native ``better-sqlite3`` is built and run under the *same* (system) Node,
    the ABI matches — so unlike a cross-runtime shim, this never falls back to
    slow WASM SQLite. Opt out with ``PUPPETMASTER_CODEGRAPH_NO_NPX=1``.
    """
    if _npx_disabled():
        return None
    npx = shutil.which("npx")
    if not npx:
        return None
    return [npx, "-y", CODEGRAPH_PACKAGE]


def codegraph_available() -> bool:
    """Return True when CodeGraph is *already* invocable with no network work.

    A cheap, side-effect-free readiness probe meant for hot paths — the
    ``doctor`` health check and per-worker context injection both gate on it.
    It counts only what resolves instantly: the ``codegraph`` shim on PATH or
    Cursor's bundled Node + the installed package.

    It deliberately does **not** count the ``npx`` fallback. npx "availability"
    means *Node exists*, not that CodeGraph is ready — the first npx run pays a
    cold download + native build. Treating that as "available" made ``doctor``
    (and every adapter) synchronously trigger a global install on any plain
    Node host, blocking the MCP server. The universal npx provisioning is
    reached only by an *explicit* CodeGraph command via :func:`run_codegraph_cli`
    / :func:`ensure_codegraph_provisioned`, where paying that cost is expected.
    """
    if shutil.which(CODEGRAPH_COMMAND) is not None:
        return True
    return _cursor_codegraph_invocation() is not None


def resolve_codegraph_invocation() -> list[str]:
    """Return the argv prefix used to invoke CodeGraph.

    Prefers running ``codegraph.js`` under Cursor's bundled Node binary
    so the runtime ABI matches whatever ``better-sqlite3`` was built
    against (see ``puppetmaster repair-codegraph``). This is the
    deterministic fix for the 18-hour runaway scenario where a stray
    ``codegraph`` invoked from a Homebrew-Node shim falls back to the
    slow WASM SQLite driver because better-sqlite3 was built for
    Cursor's Node ABI.

    Order of preference, falling through to the next on miss:

    1. ``PUPPETMASTER_CODEGRAPH_NODE`` + ``PUPPETMASTER_CODEGRAPH_JS``
       envs (escape hatch for non-standard installs).
    2. Cursor's bundled Node + the global ``@colbymchenry/codegraph``
       JS entrypoint resolved via ``npm root -g``.
    3. The bare ``codegraph`` shim on PATH (a global install — fast, no
       per-call resolution).
    4. The universal ``npx @colbymchenry/codegraph`` fallback, so any host
       with Node can run CodeGraph with zero manual setup.

    Only when none of the above resolve (no Node at all) do we return the bare
    ``codegraph`` command, whose "not found" failure surfaces the install hint.
    """
    env_node = os.environ.get("PUPPETMASTER_CODEGRAPH_NODE")
    env_js = os.environ.get("PUPPETMASTER_CODEGRAPH_JS")
    if env_node and env_js and Path(env_node).is_file() and Path(env_js).is_file():
        return [env_node, env_js]

    cursor_invocation = _cursor_codegraph_invocation()
    if cursor_invocation is not None:
        return cursor_invocation

    if shutil.which(CODEGRAPH_COMMAND) is not None:
        return [CODEGRAPH_COMMAND]

    npx_invocation = _npx_codegraph_invocation()
    if npx_invocation is not None:
        return npx_invocation

    return [CODEGRAPH_COMMAND]


# Resolving the Cursor-Node invocation shells out (`npm root -g`, filesystem
# probes) and is hit on every codegraph_available() check. The install location
# is stable for the life of the process, so memoize it. _UNSET distinguishes
# "not computed yet" from a cached negative (None) result.
_UNSET = object()
_CURSOR_INVOCATION_CACHE: Any = _UNSET


def reset_cursor_codegraph_invocation_cache() -> None:
    """Clear the memoized Cursor-Node invocation (used by tests)."""
    global _CURSOR_INVOCATION_CACHE
    _CURSOR_INVOCATION_CACHE = _UNSET


def _cursor_codegraph_invocation() -> Optional[list[str]]:
    """Return ``[cursor_node, codegraph.js]`` when both are discoverable (memoized)."""
    global _CURSOR_INVOCATION_CACHE
    if _CURSOR_INVOCATION_CACHE is not _UNSET:
        return _CURSOR_INVOCATION_CACHE
    result = _compute_cursor_codegraph_invocation()
    _CURSOR_INVOCATION_CACHE = result
    return result


def _compute_cursor_codegraph_invocation() -> Optional[list[str]]:
    """Return ``[cursor_node, codegraph.js]`` when both are discoverable."""
    # Imports deferred to avoid module-import-time cost in code paths
    # that never actually invoke codegraph (e.g. pure Puppetmaster swarm runs).
    try:
        from puppetmaster.codegraph_repair import (
            find_codegraph_install,
            find_cursor_node,
        )
    except Exception:
        return None
    node = find_cursor_node()
    if node is None:
        return None
    install = find_codegraph_install()
    if install is None:
        return None
    js = install / "dist" / "bin" / "codegraph.js"
    if not js.is_file():
        js = install / "bin" / "codegraph.js"
    if not js.is_file():
        return None
    return [str(node), str(js)]


# One-time, best-effort provisioning so CodeGraph is genuinely "always there"
# on any Node host — not just resolvable, but warm enough that the first timed
# call doesn't lose a race against a cold npx download / native build.
_PROVISION_LOCK = threading.Lock()
_PROVISION_DONE = False


def reset_codegraph_provisioning_state() -> None:
    """Clear the once-guard (used by tests)."""
    global _PROVISION_DONE
    _PROVISION_DONE = False


def _global_install_disabled() -> bool:
    raw = os.environ.get("PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _provision_timeout_seconds() -> int:
    raw = os.environ.get("PUPPETMASTER_CODEGRAPH_PROVISION_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(30, int(float(raw)))
        except ValueError:
            pass
    return 300


def ensure_codegraph_provisioned() -> bool:
    """Make CodeGraph invocable on this host, persistently, on first use.

    Idempotent (once per process) and fully best-effort — every failure mode
    falls through to "use whatever's resolvable" rather than raising. Strategy:

    * If a fast path already resolves (global ``codegraph`` shim, Cursor's
      bundled Node, or the ``PUPPETMASTER_CODEGRAPH_*`` env override) there is
      nothing to do.
    * Otherwise, when ``npm`` is present, do a one-time global install
      (``npm install -g @colbymchenry/codegraph``) so future calls hit a real
      shim with no per-call ``npx`` overhead, this session and every session
      after. Opt out with ``PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL=1``.
    * If the global install can't run or doesn't land on PATH, warm the ``npx``
      cache once (the native build happens here, off the timed call path) so the
      universal fallback is fast.

    Returns True when CodeGraph is invocable afterward. The only False is a host
    with no Node at all — a Node CLI cannot run without Node.
    """
    global _PROVISION_DONE
    if (
        shutil.which(CODEGRAPH_COMMAND) is not None
        or _cursor_codegraph_invocation() is not None
    ):
        return True
    env_node = os.environ.get("PUPPETMASTER_CODEGRAPH_NODE")
    env_js = os.environ.get("PUPPETMASTER_CODEGRAPH_JS")
    if env_node and env_js and Path(env_node).is_file() and Path(env_js).is_file():
        return True

    npx_invocation = _npx_codegraph_invocation()
    if npx_invocation is None:
        return False  # no Node — the one floor we cannot cross

    if _PROVISION_DONE:
        return True
    with _PROVISION_LOCK:
        if _PROVISION_DONE:
            return True
        _PROVISION_DONE = True

        if not _global_install_disabled():
            npm = shutil.which("npm")
            if npm is not None:
                try:
                    completed = subprocess.run(
                        [npm, "install", "-g", CODEGRAPH_PACKAGE],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=_provision_timeout_seconds(),
                        check=False,
                    )
                    if (
                        completed.returncode == 0
                        and shutil.which(CODEGRAPH_COMMAND) is not None
                    ):
                        return True
                except (OSError, subprocess.SubprocessError):
                    pass  # fall through to npx warm

        # Warm the npx cache so the first real (timed) call doesn't pay the
        # cold download + native build. Exit code is irrelevant — the point is
        # the fetch+build side effect.
        try:
            subprocess.run(
                npx_invocation + ["--help"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_provision_timeout_seconds(),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return True


def codegraph_initialized(cwd: Union[Path, str, None]) -> bool:
    """Return True when the target workspace has a .codegraph/ directory."""
    if not cwd:
        return False
    return (Path(cwd) / ".codegraph").exists()


def codegraph_ready(cwd: Union[Path, str, None]) -> bool:
    return codegraph_available() and codegraph_initialized(cwd)


def codegraph_context(
    task: str,
    cwd: Union[Path, str, None],
    *,
    max_nodes: int = 15,
    timeout_seconds: int = DEFAULT_CONTEXT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Return task-relevant CodeGraph context for the workspace, or None."""
    if not codegraph_ready(cwd):
        return None
    started = time.monotonic()
    try:
        completed = subprocess.run(
            resolve_codegraph_invocation()
            + [
                "context",
                task,
                "--max-nodes",
                str(max_nodes),
                "--format",
                "markdown",
            ],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or "").strip()
    if not output:
        return None
    output = output[:MAX_CONTEXT_CHARS]
    _record_codegraph_usage(
        cli_args=["context", task],
        cwd=str(cwd or ""),
        stdout=output,
        latency_ms=(time.monotonic() - started) * 1000.0,
        ok=True,
        caller="swarm",
    )
    return output


def puppetmaster_source_root() -> str:
    """Absolute path to the directory containing the ``puppetmaster`` package."""
    return str(Path(__file__).resolve().parents[1])


def inject_worker_cli_env(env: dict[str, str]) -> dict[str, str]:
    """Ensure a worker subprocess can self-serve the CodeGraph CLI (#4).

    A spawned agent that runs ``python -m puppetmaster codegraph ...`` may pick
    up a *stale pip install* that predates the codegraph subcommand and fail
    with "unknown command". Prepending this install's source root to PYTHONPATH
    makes the current tree shadow any older one, so the worker's CLI matches the
    parent's. Mutates and returns ``env``."""
    source_root = puppetmaster_source_root()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing}" if existing else source_root
    )
    return env


# Python env vars that select an interpreter's import roots. They are correct
# for ``python -m puppetmaster ...`` children (same interpreter as the parent),
# but poison for a worker that spawns a *foreign* Python.
_FOREIGN_PYTHON_ENV_VARS = ("PYTHONPATH", "PYTHONHOME")


def scrub_foreign_interpreter_env(env: dict[str, str]) -> dict[str, str]:
    """Strip Python import-root vars before handing ``env`` to a foreign Python.

    Puppetmaster's parent process runs under its own interpreter (e.g. a pyenv
    3.9 install) with ``PYTHONPATH`` pointed at this source tree — correct for
    its own ``python -m puppetmaster`` children. But an adapter that spawns a
    *different* Python CLI (e.g. the Hermes 3.11 binary) inherits the same env
    and imports the parent's site-packages first, which is a cross-interpreter
    version clash (observed: ``load_dotenv() got an unexpected keyword argument
    'override'`` from a stale python-dotenv). Node-based adapters (Cursor /
    Claude / Codex) ignore ``PYTHONPATH``, so a Python-CLI adapter is the first
    to hit this; any future one wants the same scrub. Mutates and returns
    ``env``."""
    for var in _FOREIGN_PYTHON_ENV_VARS:
        env.pop(var, None)
    return env


def codegraph_prompt_section(context: str) -> str:
    """Format a CodeGraph context string for prompt injection."""
    return "\n".join(
        [
            "",
            "Shared CodeGraph context for this task:",
            "```",
            context.strip(),
            "```",
            "Use these symbols and files as authoritative starting points. "
            "Confirm with the live repo before relying on them, but do not "
            "re-scan the whole codebase if CodeGraph already located the "
            "relevant area.",
            "",
            "This snapshot is a one-time view captured before you started. To "
            "refresh or expand it on demand -- e.g. to trace callers, reverse "
            "dependencies, or the blast radius of a change -- run the ABI-safe "
            "CodeGraph CLI from the repo root instead of falling back to a "
            "whole-repo grep:",
            "  python -m puppetmaster codegraph search '<symbol or keyword>'",
            "  python -m puppetmaster codegraph context '<task>' --max-nodes 15",
            "  python -m puppetmaster codegraph affected <path>",
            "If that command is unavailable in this environment it will simply "
            "error; fall back to ripgrep/git as usual.",
            "",
        ]
    )


def enrich_prompt_with_codegraph(
    prompt: str,
    *,
    task_description: str,
    cwd: Union[Path, str, None],
    disabled: bool = False,
    max_nodes: int = 15,
) -> tuple[str, bool]:
    """Append CodeGraph context to a prompt when available.

    Returns the (possibly enriched) prompt and a flag indicating whether
    CodeGraph context was actually injected.
    """
    if disabled:
        return prompt, False
    context = codegraph_context(task_description, cwd, max_nodes=max_nodes)
    if not context:
        return prompt, False
    return prompt + codegraph_prompt_section(context), True


def run_codegraph_cli(
    cli_args: list[str],
    cwd: Union[Path, str, None],
    *,
    require_initialized: bool = True,
    timeout_seconds: Optional[int] = DEFAULT_CONTEXT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a codegraph CLI subcommand and return a JSON-serializable result.

    The result always contains ``ok`` (bool), ``command`` (str), and ``cwd`` (str).
    When the CLI cannot be invoked, ``error`` describes the issue. When it
    runs, ``returncode``, ``stdout``, and ``stderr`` are included.
    """
    rendered_command = "codegraph " + " ".join(cli_args)
    cwd_str = str(cwd) if cwd else ""

    # Provision on first use (global install or npx warm) so the timed call
    # below isn't racing a cold download/build. This is also the real
    # invocability gate: it returns False only on a host with no Node at all
    # (the one floor a Node CLI can't cross), and True when the shim, Cursor's
    # Node, an env override, or the npx fallback can run CodeGraph. Unlike the
    # cheap codegraph_available() probe (used on hot paths like doctor), it
    # accounts for npx — so explicit `codegraph` commands work on any Node host.
    if not ensure_codegraph_provisioned():
        return {
            "ok": False,
            "command": rendered_command,
            "cwd": cwd_str,
            "error": CODEGRAPH_MISSING_HINT,
        }
    if require_initialized and not codegraph_initialized(cwd):
        return {
            "ok": False,
            "command": rendered_command,
            "cwd": cwd_str,
            "error": CODEGRAPH_NOT_INITIALIZED_HINT,
        }

    result = _run_codegraph_once(
        cli_args, cwd_str, rendered_command, timeout_seconds
    )

    # Auto-heal: a non-zero exit whose output matches the better-sqlite3 Node
    # ABI failure means the native binding was built for a different Node than
    # the one invoking it. Rebuild it against Cursor's Node once, then retry —
    # but only if the rebuild actually succeeded (a failed rebuild means a
    # blind retry would just reproduce the same ABI error).
    if _codegraph_should_autoheal(result):
        repair = _attempt_codegraph_autoheal()
        if repair is not None:
            if repair.get("ok"):
                result = _run_codegraph_once(
                    cli_args, cwd_str, rendered_command, timeout_seconds
                )
            result["autoheal"] = repair
    return result


def _nonpaging_env() -> dict[str, str]:
    """Environment for child CLIs that disables interactive pagers when our own
    stdout isn't a TTY (or the user set ``PUPPETMASTER_NO_PAGER``).

    A pager (git/codegraph spawning ``less``) blocks forever in a
    non-interactive shell waiting for a keypress that never comes. Forcing
    ``PAGER``/``GIT_PAGER`` to ``cat`` keeps output flowing. A TTY caller is left
    alone so interactive use still pages."""
    env = dict(os.environ)
    forced = os.environ.get("PUPPETMASTER_NO_PAGER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if forced or not sys.stdout.isatty():
        env.setdefault("PAGER", "cat")
        env["GIT_PAGER"] = "cat"
        env["PAGER"] = "cat"
    return env


def _run_codegraph_once(
    cli_args: list[str],
    cwd_str: str,
    rendered_command: str,
    timeout_seconds: Optional[int],
) -> dict[str, Any]:
    """Invoke codegraph once via the resolved (Cursor-Node) invocation."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            resolve_codegraph_invocation() + cli_args,
            cwd=cwd_str or None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=_nonpaging_env(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_stream(exc.stdout)
        stderr = _decode_stream(exc.stderr)
        return {
            "ok": False,
            "command": rendered_command,
            "cwd": cwd_str,
            "error": f"codegraph command timed out after {timeout_seconds}s",
            "stdout": stdout,
            "stderr": stderr,
        }
    except OSError as exc:
        return {
            "ok": False,
            "command": rendered_command,
            "cwd": cwd_str,
            "error": f"failed to invoke codegraph: {exc}",
        }

    _record_codegraph_usage(
        cli_args=cli_args,
        cwd=cwd_str,
        stdout=completed.stdout or "",
        latency_ms=(time.monotonic() - started) * 1000.0,
        ok=completed.returncode == 0,
        caller="mcp",
    )
    return {
        "ok": completed.returncode == 0,
        "command": rendered_command,
        "cwd": cwd_str,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


# Auto-heal coordination. A rebuild is expensive (npm) and concurrent callers
# (the MCP server runs codegraph queries on worker threads) must not each spawn
# their own `npm rebuild`. The lock makes the decision-to-rebuild atomic; the
# cooldown keeps a *failed* rebuild from wedging the process forever while still
# preventing a tight retry storm. A *successful* rebuild latches `succeeded` so
# we never rebuild again for the life of the process.
_AUTOHEAL_LOCK = threading.Lock()
_AUTOHEAL_COOLDOWN_SECONDS = 300.0
_AUTOHEAL_STATE: dict[str, Any] = {
    "succeeded": False,
    "in_progress": False,
    "last_attempt_at": 0.0,
}


def reset_codegraph_autoheal_state() -> None:
    """Reset auto-heal bookkeeping (used by tests and after a manual repair)."""
    with _AUTOHEAL_LOCK:
        _AUTOHEAL_STATE["succeeded"] = False
        _AUTOHEAL_STATE["in_progress"] = False
        _AUTOHEAL_STATE["last_attempt_at"] = 0.0


def codegraph_autoheal_enabled() -> bool:
    """Auto-heal is on by default; opt out with PUPPETMASTER_CODEGRAPH_AUTOHEAL=0."""
    return os.environ.get("PUPPETMASTER_CODEGRAPH_AUTOHEAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _codegraph_env_install_pinned() -> bool:
    """True when the user pinned a custom Node+JS install via env overrides.

    In that case ``repair_codegraph_sqlite`` (which rebuilds the *global* npm
    install) can't safely target their install, so we must not auto-heal.
    """
    node = os.environ.get("PUPPETMASTER_CODEGRAPH_NODE")
    js = os.environ.get("PUPPETMASTER_CODEGRAPH_JS")
    return bool(node and js and Path(node).is_file() and Path(js).is_file())


# Unambiguous strings emitted only by a better-sqlite3 native ABI load failure.
# Kept deliberately strict (vs. the broader `codegraph_native_sqlite_broken`
# backend detector) so a transient/unrelated non-zero exit never triggers an
# expensive npm rebuild.
_ABI_LOAD_FAILURE_PHRASES = (
    "node_module_version",
    "was compiled against a different node",
    "different node.js version",
)


def _codegraph_abi_load_failure(text: str) -> bool:
    """True only for the hard better-sqlite3 Node ABI native-load signature."""
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ABI_LOAD_FAILURE_PHRASES)


def _codegraph_should_autoheal(result: dict[str, Any]) -> bool:
    """True when a result looks like a better-sqlite3 Node ABI failure we can fix."""
    if result.get("ok"):
        return False
    if not codegraph_autoheal_enabled():
        return False
    if _codegraph_env_install_pinned():
        return False
    combined = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
    return _codegraph_abi_load_failure(combined)


def _claim_codegraph_autoheal() -> bool:
    """Atomically decide whether *this* caller should run the rebuild.

    Returns True for exactly one caller at a time, never after a success, and
    never within the cooldown window of a prior (failed) attempt.
    """
    with _AUTOHEAL_LOCK:
        if _AUTOHEAL_STATE["succeeded"] or _AUTOHEAL_STATE["in_progress"]:
            return False
        now = time.monotonic()
        last = _AUTOHEAL_STATE["last_attempt_at"]
        if last and now - last < _AUTOHEAL_COOLDOWN_SECONDS:
            return False
        _AUTOHEAL_STATE["in_progress"] = True
        _AUTOHEAL_STATE["last_attempt_at"] = now
        return True


def _attempt_codegraph_autoheal() -> Optional[dict[str, Any]]:
    """Rebuild better-sqlite3 against Cursor's Node. Returns a summary dict, or
    None when another thread already owns the (one-at-a-time) attempt or the
    cooldown/success latch forbids a new one."""
    if not _claim_codegraph_autoheal():
        return None
    summary: dict[str, Any] = {"ok": False, "message": "auto-heal failed: unknown"}
    try:
        from puppetmaster.codegraph_repair import repair_codegraph_sqlite

        result = repair_codegraph_sqlite(verify=False)
        summary = {"ok": bool(result.ok), "message": result.message}
    except Exception as exc:  # pragma: no cover - defensive
        summary = {"ok": False, "message": f"auto-heal failed: {exc}"}
    finally:
        with _AUTOHEAL_LOCK:
            _AUTOHEAL_STATE["in_progress"] = False
            if summary.get("ok"):
                _AUTOHEAL_STATE["succeeded"] = True
    return summary


def _record_codegraph_usage(
    *,
    cli_args: list[str],
    cwd: str,
    stdout: str,
    latency_ms: float,
    ok: bool,
    caller: str,
) -> None:
    """Best-effort: log this query to the global codegraph usage log. The
    command is the first CLI arg (``context``/``search``/``affected``/``files``);
    the query argument's length is recorded as a size signal, never its text."""
    try:
        from puppetmaster import codegraph_usage

        command = cli_args[0] if cli_args else ""
        query_chars = len(cli_args[1]) if len(cli_args) > 1 else 0
        codegraph_usage.record_query(
            command=command,
            cwd=cwd,
            result_chars=len(stdout or ""),
            latency_ms=latency_ms,
            ok=ok,
            caller=caller,
            query_chars=query_chars,
        )
    except Exception:
        pass


def codegraph_query(
    search: str,
    cwd: Union[Path, str, None],
    *,
    kind: Optional[str] = None,
    limit: Optional[int] = None,
    json_output: bool = True,
    timeout_seconds: int = DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph query` to find symbols by name."""
    if not search or not search.strip():
        return {
            "ok": False,
            "command": "codegraph query",
            "cwd": str(cwd or ""),
            "error": "search term is required",
        }
    args = ["query", search]
    if kind:
        args.extend(["--kind", str(kind)])
    if limit is not None:
        args.extend(["--limit", str(int(limit))])
    if json_output:
        args.append("--json")
    return run_codegraph_cli(args, cwd, timeout_seconds=timeout_seconds)


def codegraph_files_listing(
    cwd: Union[Path, str, None],
    *,
    path: Optional[str] = None,
    fmt: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    max_depth: Optional[int] = None,
    json_output: bool = True,
    timeout_seconds: int = DEFAULT_FILES_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph files` to inspect the indexed file structure."""
    args = ["files"]
    if path:
        args.append(str(path))
    if fmt:
        args.extend(["--format", str(fmt)])
    if filter_pattern:
        args.extend(["--filter", str(filter_pattern)])
    if max_depth is not None:
        args.extend(["--max-depth", str(int(max_depth))])
    if json_output:
        args.append("--json")
    return run_codegraph_cli(args, cwd, timeout_seconds=timeout_seconds)


def codegraph_affected(
    files: list[str],
    cwd: Union[Path, str, None],
    *,
    depth: Optional[int] = None,
    filter_pattern: Optional[str] = None,
    json_output: bool = True,
    timeout_seconds: int = DEFAULT_AFFECTED_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph affected` to find tests impacted by changed files."""
    if not files:
        return {
            "ok": False,
            "command": "codegraph affected",
            "cwd": str(cwd or ""),
            "error": "at least one changed file path is required",
        }
    args = ["affected"]
    args.extend(str(item) for item in files)
    if depth is not None:
        args.extend(["--depth", str(int(depth))])
    if filter_pattern:
        args.extend(["--filter", str(filter_pattern)])
    if json_output:
        args.append("--json")
    return run_codegraph_cli(args, cwd, timeout_seconds=timeout_seconds)


def codegraph_context_command(
    task: str,
    cwd: Union[Path, str, None],
    *,
    max_nodes: int = 15,
    fmt: str = "markdown",
    timeout_seconds: int = DEFAULT_CONTEXT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph context` and return the raw CLI payload."""
    if not task or not task.strip():
        return {
            "ok": False,
            "command": "codegraph context",
            "cwd": str(cwd or ""),
            "error": "task description is required",
        }
    args = [
        "context",
        task,
        "--max-nodes",
        str(int(max_nodes)),
        "--format",
        str(fmt),
    ]
    return run_codegraph_cli(args, cwd, timeout_seconds=timeout_seconds)


def codegraph_status_command(
    cwd: Union[Path, str, None],
    *,
    timeout_seconds: int = DEFAULT_STATUS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph status` to inspect index health."""
    return run_codegraph_cli(
        ["status"],
        cwd,
        require_initialized=False,
        timeout_seconds=timeout_seconds,
    )


def codegraph_init_command(
    cwd: Union[Path, str, None],
    *,
    index: bool = False,
    timeout_seconds: int = DEFAULT_INIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run `codegraph init` (optionally indexing immediately)."""
    args = ["init"]
    if index:
        args.append("--index")
    return run_codegraph_cli(
        args,
        cwd,
        require_initialized=False,
        timeout_seconds=timeout_seconds,
    )


def _decode_stream(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        try:
            return stream.decode()
        except UnicodeDecodeError:
            return stream.decode(errors="replace")
    return str(stream)


# --- Per-repo indexer lock --------------------------------------------------
#
# Each repo's CodeGraph index lives in its own SQLite database at
# ``<repo>/.codegraph/codegraph.db``, so two indexers on two **different**
# repos can never trash each other's data. The original implementation
# took out one machine-wide lock to defend against a much narrower
# concern: two indexers on the **same** repo (or two against the same
# DB file). That over-served the requirement and serialized unrelated
# repos for no real reason — users with 3 repos open in 3 chats kept
# hitting "Another CodeGraph indexer is already running" even when the
# repos had nothing to do with each other.
#
# v0.5.5 keys the lock by the resolved repo root path hash, so two
# repos run in parallel and the lock only blocks legitimate overlap on
# the same SQLite DB. Stale-PID auto-clear handles the case where a
# previous indexer died ungracefully and left an advisory ``flock``
# orphan (rare, but observable when ``codegraph init --index`` is
# killed via ``kill -9``).


def codegraph_lock_path(repo_root: Optional[Union[Path, str]] = None) -> Path:
    """Return the per-repo lock file used to serialize CodeGraph indexers.

    Passing ``repo_root=None`` returns the legacy machine-wide lock path
    for backwards compatibility with callers that haven't been updated
    (tests, third-party tooling). Production callers should always pass
    the target repo's root path so different repos can index in
    parallel.
    """
    base = os.environ.get("PUPPETMASTER_CODEGRAPH_LOCK_DIR")
    if base:
        directory = Path(base)
    else:
        directory = _default_cache_root() / "puppetmaster"
    mkdir_private(directory)
    if repo_root is None:
        return directory / "codegraph-indexer.lock"
    resolved = Path(repo_root).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name).strip("-") or "repo"
    return directory / f"codegraph-indexer-{safe_name}-{digest}.lock"


def _default_cache_root() -> Path:
    """Resolve the per-user cache root, respecting XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg)
    import sys

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


class CodegraphLockBusy(RuntimeError):
    """Raised when another indexer already holds the per-repo lock.

    Pre-v0.5.5 the lock was machine-wide, so this also fired for
    unrelated repos. Now it only fires when a legitimately overlapping
    indexer is already running against the *same* SQLite DB.
    """

    def __init__(self, holder_pid: Optional[int], lock_path: Path):
        self.holder_pid = holder_pid
        self.lock_path = lock_path
        message = (
            "Another CodeGraph indexer is already running for this repo"
            + (f" (pid {holder_pid})" if holder_pid else "")
            + ". Wait for it to finish, or kill it with "
            + f"`pkill -f 'codegraph init --index'` if it is stuck. Lock file: {lock_path}."
        )
        super().__init__(message)


def acquire_codegraph_lock(
    *,
    lock_path: Optional[Path] = None,
    repo_root: Optional[Union[Path, str]] = None,
) -> "CodegraphLock":
    """Acquire the CodeGraph indexer lock for ``repo_root``.

    Returns a ``CodegraphLock`` that should be ``release()``-d (or used
    as a context manager) once the indexer terminates. Raises
    :class:`CodegraphLockBusy` immediately if another holder is active —
    we never block, because that would defeat the purpose of the
    multi-threaded MCP server.

    Either pass an explicit ``lock_path`` (legacy callers) or pass the
    target ``repo_root`` and let the helper derive the per-repo path
    via :func:`codegraph_lock_path`.
    """
    if lock_path is None:
        lock_path = codegraph_lock_path(repo_root)
    return CodegraphLock(lock_path).acquire()


class CodegraphLock:
    """File-based advisory lock for CodeGraph indexer operations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> "CodegraphLock":
        # POSIX uses a non-blocking flock; Windows (no fcntl) uses msvcrt's
        # mandatory byte-range lock. Either way the second acquirer fails fast.
        fcntl = _import_fcntl()
        msvcrt = _import_msvcrt() if fcntl is None else None
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        self._fd = open_private(self.path, flags)
        if fcntl is not None or msvcrt is not None:
            if not self._try_lock(fcntl, msvcrt):
                # Stale-PID auto-clear: if the lock file records a PID that
                # isn't alive anymore, the previous indexer died ungracefully
                # and its kernel-level lock has been released — what we
                # observed is a different holder (or macOS keeping the flag a
                # moment longer). Re-check the PID; if dead, truncate and retry
                # once. If alive, surface busy so the caller can decide to wait.
                holder_pid = _read_holder_pid(self.path)
                if holder_pid is not None and not _pid_is_alive(holder_pid):
                    os.ftruncate(self._fd, 0)
                    if not self._try_lock(fcntl, msvcrt):
                        os.close(self._fd)
                        self._fd = None
                        raise CodegraphLockBusy(
                            _read_holder_pid(self.path), self.path
                        )
                else:
                    os.close(self._fd)
                    self._fd = None
                    raise CodegraphLockBusy(holder_pid, self.path)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{os.getpid()}\n".encode("utf-8"))
        return self

    def _try_lock(self, fcntl, msvcrt) -> bool:
        """Attempt a non-blocking exclusive lock. True on success."""
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                # Windows msvcrt locks are *mandatory*: a locked byte range
                # can't even be read by other handles. We must keep the PID
                # text readable for the busy-check, so lock a single sentinel
                # byte far past any content rather than byte 0. Locking beyond
                # EOF is allowed on Windows.
                os.lseek(self._fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                os.lseek(self._fd, 0, os.SEEK_SET)
            return True
        except (OSError, BlockingIOError):
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl = _import_fcntl()
        msvcrt = _import_msvcrt() if fcntl is None else None
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            elif msvcrt is not None:
                os.lseek(self._fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                os.lseek(self._fd, 0, os.SEEK_SET)
        except OSError:
            pass
        try:
            os.close(self._fd)
        finally:
            self._fd = None

    def __enter__(self) -> "CodegraphLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _read_holder_pid(path: Path) -> Optional[int]:
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not contents.isdigit():
        return None
    return int(contents)


# A sentinel byte offset (1 GiB) used only on Windows for the msvcrt advisory
# lock. It sits far past the tiny PID text so the mandatory lock never blocks
# readers of the holder PID, while still serializing acquirers on one byte.
_WIN_LOCK_OFFSET = 1 << 30


def _import_fcntl():
    try:
        import fcntl  # type: ignore[import-not-found]

        return fcntl
    except ImportError:  # Windows
        return None


def _import_msvcrt():
    try:
        import msvcrt  # type: ignore[import-not-found]

        return msvcrt
    except ImportError:  # non-Windows
        return None


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists for this user.

    POSIX: ``os.kill(pid, 0)`` sends no signal but errors out when the pid
    doesn't exist or we lack permission. We treat "no such process" (ESRCH)
    as dead and "permission denied" (EPERM) as alive — if a different user
    owns the pid, we shouldn't blow away their lock.

    Windows: ``os.kill(pid, 0)`` does NOT raise for a missing pid (it maps to
    ``TerminateProcess`` and surfaces a generic ``OSError``), so the POSIX path
    would wrongly report every dead pid as alive — leaving a stale lock held by
    a long-gone process unbreakable. Query the OS directly via ``OpenProcess``
    and distinguish "missing pid" from "access denied" by the Win32 error code.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_is_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _pid_is_alive_windows(pid: int) -> bool:
    """Windows liveness probe via the Win32 API (no psutil dependency)."""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    ERROR_INVALID_PARAMETER = 87

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error() or kernel32.GetLastError()
        # No such process -> dead. Access denied -> someone else's live process
        # (mirror POSIX EPERM -> alive so we don't break another user's lock).
        if err == ERROR_INVALID_PARAMETER:
            return False
        if err == ERROR_ACCESS_DENIED:
            return True
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# --- Native SQLite health check ---------------------------------------------


def codegraph_native_sqlite_broken(status_output: str) -> bool:
    """Return True when `codegraph status` output suggests the native SQLite
    driver (better-sqlite3) failed to load and the WASM fallback is active.

    We pattern-match on the known surface symptoms because CodeGraph itself
    emits these strings when the native module can't be required.
    """
    if not status_output:
        return False
    lowered = status_output.lower()
    fallback_markers = (
        "wasm",
        "wasm fallback",
        "different node abi",
        "node abi",
        "better-sqlite3",
        "backend: fallback",
        "backend: wasm",
        # Hard native-load failure surface (command exits non-zero before it
        # can fall back to WASM). This is the exact error a shell-Node
        # invocation throws when better-sqlite3 was built for a different ABI.
        "node_module_version",
        "better_sqlite3",
        "was compiled against a different node",
        "different node.js version",
    )
    return any(marker in lowered for marker in fallback_markers)
