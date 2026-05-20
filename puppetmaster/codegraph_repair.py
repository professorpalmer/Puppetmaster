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


def find_cursor_node(explicit: Optional[str] = None) -> Optional[Path]:
    """Return the path to Cursor's bundled Node, or None if not found.

    Callers can pass ``explicit`` to override the search (used by the CLI's
    ``--cursor-node`` flag). The override is returned as long as the file
    exists; otherwise we walk the per-platform candidate list.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
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
    return None


def find_codegraph_install(npm_command: str = "npm") -> Optional[Path]:
    """Resolve the global CodeGraph install directory via ``npm root -g``.

    We deliberately call npm rather than guessing common paths because
    npm respects user prefix overrides (nvm, fnm, asdf, Homebrew, etc.)
    and the right answer is whatever ``npm root -g`` reports for the Node
    that owns the global CodeGraph install.
    """
    npm_path = shutil.which(npm_command)
    if not npm_path:
        return None
    try:
        completed = subprocess.run(
            [npm_path, "root", "-g"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    root = (completed.stdout or "").strip()
    if not root:
        return None
    codegraph_dir = Path(root) / "@colbymchenry" / "codegraph"
    if not codegraph_dir.is_dir():
        return None
    return codegraph_dir


def detect_node_version(node_path: Path, timeout_seconds: int = 5) -> Optional[str]:
    """Return ``node --version`` output (e.g. ``v22.22.0``) or None on failure."""
    try:
        completed = subprocess.run(
            [str(node_path), "--version"],
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
    """Run the better-sqlite3 rebuild against Cursor's Node.

    Returns a :class:`RepairResult` with a structured payload that both
    the CLI and the MCP tool render. We intentionally never raise; any
    failure becomes ``ok=False`` plus a clear message so the agent can
    surface fix-it instructions to the user.
    """
    node_path = find_cursor_node(cursor_node)
    if node_path is None:
        return RepairResult(
            ok=False,
            message=(
                "Could not find Cursor's bundled Node. Pass --cursor-node "
                "with the path explicitly. On macOS the expected location "
                "is /Applications/Cursor.app/Contents/Resources/app/resources/helpers/node."
            ),
            next_steps=[
                "Locate the Node bundled with Cursor (Cursor.app/Contents/Resources/...).",
                "Re-run with --cursor-node </path/to/cursor-node>.",
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
        "Restart the Puppetmaster MCP server in Cursor (Settings -> MCP -> toggle puppetmaster off/on).",
        "Re-run `puppetmaster_codegraph_status` against a target repo and confirm Backend: native.",
    ]
    if verify_backend and verify_backend.lower() != "native":
        next_steps.insert(
            0,
            (
                "Verification reported Backend: "
                f"{verify_backend}. Try restarting Cursor entirely so the MCP "
                "server picks up the rebuilt native module."
            ),
        )

    return RepairResult(
        ok=True,
        message=(
            f"Rebuilt better-sqlite3 for Cursor's Node {node_version or 'unknown'} "
            f"in {install_path}. Restart the Puppetmaster MCP server in Cursor "
            "to pick up the change."
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
        try:
            return stream.decode()
        except UnicodeDecodeError:
            return stream.decode(errors="replace")
    return str(stream)
