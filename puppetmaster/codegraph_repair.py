"""Repair CodeGraph's better-sqlite3 native binding for the Cursor MCP runtime.

Why this exists
---------------

CodeGraph ships a native SQLite driver (``better-sqlite3``) for speed.
That native module is locked to a specific Node ABI: the version of
``NODE_MODULE_VERSION`` baked in at install/rebuild time. If the Node
that imports it later has a different ABI, the module fails to load
and CodeGraph silently falls back to a much slower WASM driver.

The trap most Cursor users will hit:

  * Your terminal's Node (often Homebrew, e.g. v23.x with ABI 131).
  * Cursor.app's bundled Node (v22.x with ABI 127 on current builds).

A naive ``npm rebuild better-sqlite3`` in your shell builds for the
shell's Node, which is the *wrong* runtime for Puppetmaster's MCP
server (which Cursor spawns under its own bundled Node). The MCP
process then sees the "compiled against a different Node ABI" error,
falls back to WASM, and you get ``database is locked`` / ``unable to
open database file`` from concurrent indexers.

This module:

  1. Finds Cursor's bundled Node binary on the local machine.
  2. Finds the global CodeGraph install directory.
  3. Runs ``npm rebuild better-sqlite3`` with Cursor's Node first on
     PATH, so the rebuild targets the runtime Puppetmaster actually
     uses.
  4. Optionally verifies the result with ``codegraph status`` under
     the same Node.

The CLI exposes this as ``python -m puppetmaster repair-codegraph``;
the MCP server exposes ``puppetmaster_repair_codegraph`` so an agent
can self-heal when it sees the WASM-fallback error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Candidate locations for Cursor's bundled Node, in priority order.
# On macOS the helpers path is stable across Cursor versions; we also
# probe a couple of older / alternate layouts just in case.
_CURSOR_NODE_CANDIDATES_MAC = (
    "/Applications/Cursor.app/Contents/Resources/app/resources/helpers/node",
    "/Applications/Cursor.app/Contents/Frameworks/Cursor Helper.app/Contents/MacOS/node",
)

# Linux .deb / AppImage installs.
_CURSOR_NODE_CANDIDATES_LINUX = (
    "/opt/cursor/resources/app/resources/helpers/node",
    "/usr/share/cursor/resources/app/resources/helpers/node",
    str(Path.home() / ".local/share/cursor/resources/app/resources/helpers/node"),
)

# Windows install paths (we don't actively support windows in tests but
# the detection cost is zero).
_CURSOR_NODE_CANDIDATES_WIN = (
    str(Path.home() / "AppData/Local/Programs/cursor/resources/app/resources/helpers/node.exe"),
)


@dataclass
class RepairResult:
    """Outcome of a CodeGraph better-sqlite3 rebuild attempt."""

    ok: bool
    message: str
    cursor_node_path: Optional[str] = None
    cursor_node_version: Optional[str] = None
    codegraph_install_path: Optional[str] = None
    rebuild_stdout: str = ""
    rebuild_stderr: str = ""
    verify_backend: Optional[str] = None
    next_steps: list[str] = None  # type: ignore[assignment]

    def to_payload(self) -> dict:
        return {
            "ok": self.ok,
            "message": self.message,
            "cursor_node_path": self.cursor_node_path,
            "cursor_node_version": self.cursor_node_version,
            "codegraph_install_path": self.codegraph_install_path,
            "rebuild_stdout": self.rebuild_stdout,
            "rebuild_stderr": self.rebuild_stderr,
            "verify_backend": self.verify_backend,
            "next_steps": list(self.next_steps or []),
        }


def find_runtime_node(explicit: Optional[str] = None) -> Optional[Path]:
    """Return a Node binary to rebuild/run CodeGraph under, or None.

    Puppetmaster supports five harnesses (cursor, claude-code, codex, openai,
    hermes); only one of them is Cursor. The better-sqlite3 native module must
    be rebuilt against *whatever Node actually runs CodeGraph on this host*, not
    Cursor specifically. Resolution order, first hit wins:

    1. ``explicit`` — the CLI ``--cursor-node`` / ``--runtime-node`` flag.
    2. ``PUPPETMASTER_CODEGRAPH_NODE`` env override (matches the resolver in
       ``codegraph.resolve_codegraph_invocation``).
    3. Cursor's bundled Node, when Cursor is installed (kept first among the
       auto-detected options because its ABI is the historical trap this whole
       module exists to fix).
    4. Any ``node`` on PATH — the universal fallback that unblocks every
       non-Cursor harness.

    A returned path is only guaranteed to exist; ABI suitability is verified
    downstream by the post-rebuild ``codegraph status`` check.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env_node = os.environ.get("PUPPETMASTER_CODEGRAPH_NODE")
    if env_node:
        env_path = Path(env_node).expanduser()
        if env_path.is_file():
            return env_path
    candidates: tuple[str, ...]
    if sys.platform == "darwin":
        candidates = _CURSOR_NODE_CANDIDATES_MAC
    elif sys.platform.startswith("linux"):
        candidates = _CURSOR_NODE_CANDIDATES_LINUX
    elif sys.platform.startswith("win"):
        candidates = _CURSOR_NODE_CANDIDATES_WIN
    else:
        candidates = ()
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    which_node = shutil.which("node")
    if which_node:
        return Path(which_node)
    return None


# Back-compat alias: this function used to be Cursor-only. The MCP/CLI repair
# path and tests still import ``find_cursor_node``; keep the old name pointing at
# the generalized resolver so every harness benefits without an API break.
find_cursor_node = find_runtime_node


def _codegraph_install_from_shim() -> Optional[Path]:
    """Resolve the install dir by following the ``codegraph`` shim on PATH.

    ``npm root -g`` reports the global dir for *whichever npm is first on PATH*.
    When CodeGraph was installed under a different prefix (classic case: the
    package lives in Homebrew's ``/opt/homebrew/lib/node_modules`` while the
    PATH npm is a pyenv/fnm/Hermes-bundled npm pointing elsewhere), ``npm root
    -g`` misses it even though the ``codegraph`` shim is right there. The shim is
    a symlink into the real package, so resolving it backwards finds the true
    install regardless of npm prefix. The shim points at
    ``<pkg>/dist/bin/codegraph.js`` (or ``<pkg>/bin/codegraph.js``); walk up to
    the package root.
    """
    shim = shutil.which("codegraph")
    if not shim:
        return None
    try:
        target = Path(shim).resolve()
        for parent in target.parents:
            if parent.name == "codegraph" and parent.parent.name == "@colbymchenry":
                if parent.is_dir():
                    return parent
    except (OSError, RuntimeError, ValueError):
        # A malformed shim path (too long, embedded NULs, weird mocks in tests)
        # must never crash CodeGraph resolution — treat as "not found".
        return None
    return None


def _looks_like_single_path(value: str) -> bool:
    """True when ``value`` plausibly is a single filesystem path.

    ``npm root -g`` returns exactly one line: an absolute directory path. A
    misconfigured npm, a shim/alias that prints banners, or (in tests) a mocked
    ``subprocess.run`` can hand back multi-line blobs or huge JSON. Feeding that
    straight into ``Path(...).is_dir()`` raises ``OSError: File name too long``
    on most platforms, so we sanity-gate before touching the filesystem.
    """
    if not value or "\n" in value or "\x00" in value:
        return False
    # A real node_modules root is short; cap well under the OS path limit so a
    # giant blob is rejected before it can raise ENAMETOOLONG.
    return len(value) <= 4096


def find_codegraph_install(npm_command: str = "npm") -> Optional[Path]:
    """Resolve the global CodeGraph install directory.

    Resolution order, first hit wins:

    1. ``npm root -g`` for ``npm_command``. We try npm first because it respects
       user prefix overrides (nvm, fnm, asdf, Homebrew, etc.) and is the right
       answer whenever the install and the PATH npm share a prefix.
    2. Following the ``codegraph`` shim on PATH backwards to its package root.
       This catches the cross-prefix case npm misses — the package installed
       under one Node's prefix while a different npm leads PATH (observed with a
       Homebrew CodeGraph install + a pyenv/Hermes npm). Universal across
       harnesses, since every host that can run CodeGraph has the shim.
    """
    npm_path = shutil.which(npm_command)
    if npm_path:
        try:
            completed = subprocess.run(
                [npm_path, "root", "-g"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            root = (completed.stdout or "").strip()
            # Only touch the filesystem when the output actually looks like a
            # single path — a garbage/multi-line/huge result would otherwise
            # raise ENAMETOOLONG from is_dir().
            if root and _looks_like_single_path(root):
                try:
                    codegraph_dir = Path(root) / "@colbymchenry" / "codegraph"
                    if codegraph_dir.is_dir():
                        return codegraph_dir
                except (OSError, ValueError):
                    pass  # fall through to the shim resolver
    # npm root -g missed (no npm, wrong prefix, not installed there, or a
    # non-path result). Fall back to the shim, which points straight at the
    # real package.
    return _codegraph_install_from_shim()


def detect_node_version(node_path: Path, timeout_seconds: int = 5) -> Optional[str]:
    """Return ``node --version`` output (e.g. ``v22.22.0``) or None on failure."""
    try:
        completed = subprocess.run(
            [str(node_path), "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or "").strip() or None


def repair_codegraph_sqlite(
    *,
    cursor_node: Optional[str] = None,
    codegraph_install: Optional[str] = None,
    npm_command: str = "npm",
    rebuild_timeout_seconds: int = 180,
    verify: bool = True,
    verify_cwd: Optional[str] = None,
) -> RepairResult:
    """Run the better-sqlite3 rebuild against the host's CodeGraph Node.

    Targets whatever Node actually runs CodeGraph on this host (Cursor's
    bundled Node when present, else ``PUPPETMASTER_CODEGRAPH_NODE`` or the
    ``node`` on PATH) — not Cursor specifically, so the repair works across all
    supported harnesses (cursor, claude-code, codex, openai, hermes).

    Returns a :class:`RepairResult` with a structured payload that both
    the CLI and the MCP tool render. We intentionally never raise; any
    failure becomes ``ok=False`` plus a clear message so the agent can
    surface fix-it instructions to the user.
    """
    node_path = find_runtime_node(cursor_node)
    if node_path is None:
        return RepairResult(
            ok=False,
            message=(
                "Could not find a Node runtime to rebuild CodeGraph against. "
                "Install Node.js 18+ (https://nodejs.org) so `node` is on PATH, "
                "or pass --cursor-node with an explicit Node binary path. If you "
                "use Cursor, its bundled Node is at "
                "/Applications/Cursor.app/Contents/Resources/app/resources/helpers/node "
                "(macOS)."
            ),
            next_steps=[
                "Install Node.js 18+ so `node` resolves on PATH, or set "
                "PUPPETMASTER_CODEGRAPH_NODE to a Node binary.",
                "Alternatively re-run with --cursor-node </path/to/node> "
                "pointing at the Node your harness runs CodeGraph under.",
            ],
        )

    node_version = detect_node_version(node_path)

    if codegraph_install:
        install_path = Path(codegraph_install).expanduser()
        if not install_path.is_dir():
            return RepairResult(
                ok=False,
                message=(
                    f"--codegraph-install path does not exist: {install_path}"
                ),
                cursor_node_path=str(node_path),
                cursor_node_version=node_version,
            )
    else:
        resolved = find_codegraph_install(npm_command=npm_command)
        if resolved is None:
            return RepairResult(
                ok=False,
                message=(
                    "Could not locate the global @colbymchenry/codegraph install. "
                    "Run `npm install -g @colbymchenry/codegraph` first, or pass "
                    "--codegraph-install with the full path to its install directory."
                ),
                cursor_node_path=str(node_path),
                cursor_node_version=node_version,
                next_steps=[
                    "Install CodeGraph globally: npm install -g @colbymchenry/codegraph",
                    "Then re-run `python -m puppetmaster repair-codegraph`.",
                ],
            )
        install_path = resolved

    npm_path = shutil.which(npm_command)
    if not npm_path:
        return RepairResult(
            ok=False,
            message=(
                "npm not found on PATH. Install Node.js (which ships npm) and re-run."
            ),
            cursor_node_path=str(node_path),
            cursor_node_version=node_version,
            codegraph_install_path=str(install_path),
        )

    env = os.environ.copy()
    env["PATH"] = f"{node_path.parent}{os.pathsep}{env.get('PATH', '')}"

    try:
        rebuild = subprocess.run(
            [npm_path, "rebuild", "better-sqlite3"],
            cwd=str(install_path),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=rebuild_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RepairResult(
            ok=False,
            message=(
                f"`npm rebuild better-sqlite3` timed out after {rebuild_timeout_seconds}s. "
                "Try running the command directly in a shell with PATH preset to Cursor's Node."
            ),
            cursor_node_path=str(node_path),
            cursor_node_version=node_version,
            codegraph_install_path=str(install_path),
            rebuild_stdout=_decode_stream(exc.stdout),
            rebuild_stderr=_decode_stream(exc.stderr),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RepairResult(
            ok=False,
            message=f"Failed to launch `npm rebuild better-sqlite3`: {exc}",
            cursor_node_path=str(node_path),
            cursor_node_version=node_version,
            codegraph_install_path=str(install_path),
        )

    if rebuild.returncode != 0:
        return RepairResult(
            ok=False,
            message=(
                "`npm rebuild better-sqlite3` exited non-zero. Review the stderr "
                "below — most commonly this means node-gyp can't find Python or "
                "a build toolchain (Xcode CLT on macOS, build-essential on Linux)."
            ),
            cursor_node_path=str(node_path),
            cursor_node_version=node_version,
            codegraph_install_path=str(install_path),
            rebuild_stdout=rebuild.stdout or "",
            rebuild_stderr=rebuild.stderr or "",
        )

    verify_backend: Optional[str] = None
    if verify:
        verify_backend = _verify_native_backend(
            node_path=node_path,
            install_path=install_path,
            target_cwd=verify_cwd,
        )

    next_steps = [
        "Restart the Puppetmaster MCP server in your harness so it reloads the "
        "rebuilt native module (Cursor: Settings -> MCP -> toggle puppetmaster "
        "off/on; Claude Code / Codex / Hermes: restart the session or "
        "reconnect the MCP).",
        "Re-run `puppetmaster_codegraph_status` against a target repo and confirm Backend: native.",
    ]
    if verify_backend and verify_backend.lower() != "native":
        next_steps.insert(
            0,
            (
                "Verification reported Backend: "
                f"{verify_backend}. Try restarting your harness entirely so the "
                "MCP server picks up the rebuilt native module."
            ),
        )

    return RepairResult(
        ok=True,
        message=(
            f"Rebuilt better-sqlite3 for Node {node_version or 'unknown'} "
            f"in {install_path}. Restart the Puppetmaster MCP server in your "
            "harness to pick up the change."
        ),
        cursor_node_path=str(node_path),
        cursor_node_version=node_version,
        codegraph_install_path=str(install_path),
        rebuild_stdout=rebuild.stdout or "",
        rebuild_stderr=rebuild.stderr or "",
        verify_backend=verify_backend,
        next_steps=next_steps,
    )


def _verify_native_backend(
    *,
    node_path: Path,
    install_path: Path,
    target_cwd: Optional[str],
    timeout_seconds: int = 30,
) -> Optional[str]:
    """Run `codegraph status` under Cursor's Node and parse the Backend line.

    We intentionally invoke CodeGraph's JS entry point through the same
    Node binary the MCP server uses, so the verification matches the
    real runtime — running ``codegraph status`` via the shim on PATH
    would invoke whatever Node owns the shim, which is the source of
    the original confusion.
    """
    entry = install_path / "dist" / "bin" / "codegraph.js"
    if not entry.is_file():
        entry = install_path / "bin" / "codegraph.js"
    if not entry.is_file():
        return None
    try:
        completed = subprocess.run(
            [str(node_path), str(entry), "status"],
            cwd=target_cwd or None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    for line in output.splitlines():
        stripped = line.strip()
        # Match "Backend: native" / "Backend:   native" / ANSI-coloured variants.
        if "Backend" in stripped and ":" in stripped:
            value = stripped.split(":", 1)[1].strip()
            # Strip simple ANSI colour codes so callers see plain text.
            cleaned = _strip_ansi(value).strip()
            if cleaned:
                return cleaned
    return None


def _strip_ansi(value: str) -> str:
    """Remove SGR ANSI escape sequences from a string."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _decode_stream(stream) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
