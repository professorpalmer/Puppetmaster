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

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Union


CODEGRAPH_COMMAND = "codegraph"
MAX_CONTEXT_CHARS = 4000
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 30
DEFAULT_STATUS_TIMEOUT_SECONDS = 10
DEFAULT_QUERY_TIMEOUT_SECONDS = 15
DEFAULT_AFFECTED_TIMEOUT_SECONDS = 30
DEFAULT_FILES_TIMEOUT_SECONDS = 15
DEFAULT_INIT_TIMEOUT_SECONDS = 600


CODEGRAPH_MISSING_HINT = (
    "codegraph CLI not on PATH. Install with `npm install -g @colbymchenry/codegraph` "
    "or run `npx @colbymchenry/codegraph` once to set it up."
)
CODEGRAPH_NOT_INITIALIZED_HINT = (
    "workspace is not initialized for CodeGraph. Run `puppetmaster_codegraph_init` "
    "or `codegraph init` in the target repository first."
)


def codegraph_available() -> bool:
    """Return True when the codegraph CLI is on PATH."""
    return shutil.which(CODEGRAPH_COMMAND) is not None


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
    try:
        completed = subprocess.run(
            [
                CODEGRAPH_COMMAND,
                "context",
                task,
                "--max-nodes",
                str(max_nodes),
                "--format",
                "markdown",
            ],
            cwd=str(cwd),
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
    return output[:MAX_CONTEXT_CHARS]


def codegraph_status_line(
    cwd: Union[Path, str, None],
    *,
    timeout_seconds: int = DEFAULT_STATUS_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Return a short codegraph status string for diagnostics, or None."""
    if not codegraph_ready(cwd):
        return None
    try:
        completed = subprocess.run(
            [CODEGRAPH_COMMAND, "status"],
            cwd=str(cwd),
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
    timeout_seconds: int = DEFAULT_CONTEXT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a codegraph CLI subcommand and return a JSON-serializable result.

    The result always contains ``ok`` (bool), ``command`` (str), and ``cwd`` (str).
    When the CLI cannot be invoked, ``error`` describes the issue. When it
    runs, ``returncode``, ``stdout``, and ``stderr`` are included.
    """
    rendered_command = "codegraph " + " ".join(cli_args)
    cwd_str = str(cwd) if cwd else ""

    if not codegraph_available():
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

    try:
        completed = subprocess.run(
            [CODEGRAPH_COMMAND] + cli_args,
            cwd=cwd_str or None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
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

    return {
        "ok": completed.returncode == 0,
        "command": rendered_command,
        "cwd": cwd_str,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


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
