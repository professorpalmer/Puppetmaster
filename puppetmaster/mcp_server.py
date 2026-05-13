from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.codegraph import (
    codegraph_affected,
    codegraph_context_command,
    codegraph_files_listing,
    codegraph_init_command,
    codegraph_query,
    codegraph_status_command,
)
from puppetmaster.state import resolve_state_dir


JsonObject = dict[str, Any]
ASYNC_PROCESSES: list[subprocess.Popen] = []


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: JsonObject
    handler: Callable[[JsonObject], JsonObject]


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_message(json.loads(line))
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


def handle_message(message: JsonObject) -> Optional[JsonObject]:
    method = message.get("method")
    request_id = message.get("id")

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "puppetmaster", "version": "0.2.0-beta.1"},
            }
        elif method == "tools/list":
            result = {"tools": [tool_to_json(tool) for tool in tools()]}
        elif method == "tools/call":
            params = message.get("params") or {}
            result = call_tool(str(params.get("name", "")), params.get("arguments") or {})
        else:
            return error_response(request_id, -32601, f"Unknown MCP method: {method}")
    except Exception as exc:
        return error_response(request_id, -32000, str(exc))

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def call_tool(name: str, arguments: JsonObject) -> JsonObject:
    registry = {tool.name: tool for tool in tools()}
    tool = registry.get(name)
    if tool is None:
        raise ValueError(f"Unknown Puppetmaster tool: {name}")
    return tool.handler(arguments)


def tools() -> list[McpTool]:
    return [
        McpTool(
            name="puppetmaster_doctor",
            description="Check Puppetmaster runtime, SQLite state, and provider adapter setup.",
            input_schema=base_schema(),
            handler=lambda args: run_cli(["doctor"], args),
        ),
        McpTool(
            name="puppetmaster_cursor_review",
            description="Run a Cursor SDK review worker through Puppetmaster and wait for completion.",
            input_schema=goal_schema(
                "Review this repo and identify risks, findings, and verification gaps."
            ),
            handler=lambda args: run_cursor(args, review=True),
        ),
        McpTool(
            name="puppetmaster_start_cursor_review",
            description="Start a Cursor SDK review worker asynchronously and return job_id immediately.",
            input_schema=goal_schema(
                "Review this repo and identify risks, findings, and verification gaps."
            ),
            handler=lambda args: start_cursor(args, review=True),
        ),
        McpTool(
            name="puppetmaster_cursor_plan",
            description="Run a Cursor SDK planning worker through Puppetmaster and wait for completion.",
            input_schema=goal_schema("Plan the next safe implementation slice for this repo."),
            handler=lambda args: run_cursor(args, plan=True),
        ),
        McpTool(
            name="puppetmaster_start_cursor_plan",
            description="Start a Cursor SDK planning worker asynchronously and return job_id immediately.",
            input_schema=goal_schema("Plan the next safe implementation slice for this repo."),
            handler=lambda args: start_cursor(args, plan=True),
        ),
        McpTool(
            name="puppetmaster_claude_implement",
            description="Run Claude Code as a full-edit Puppetmaster worker and wait for completion.",
            input_schema=claude_schema(),
            handler=run_claude,
        ),
        McpTool(
            name="puppetmaster_start_claude_implement",
            description="Start Claude Code as a full-edit worker asynchronously and return job_id immediately.",
            input_schema=claude_schema(),
            handler=start_claude,
        ),
        McpTool(
            name="puppetmaster_start_swarm",
            description="Start a local Puppetmaster swarm asynchronously and return job_id immediately.",
            input_schema=swarm_schema(),
            handler=start_swarm,
        ),
        McpTool(
            name="puppetmaster_start_cursor_swarm",
            description="Start a multi-role Cursor SDK analysis swarm asynchronously and return job_id immediately.",
            input_schema=cursor_swarm_schema(),
            handler=start_cursor_swarm,
        ),
        McpTool(
            name="puppetmaster_last_job",
            description="Return the most recent Puppetmaster job id.",
            input_schema=base_schema(),
            handler=lambda args: run_cli(["last"], args),
        ),
        McpTool(
            name="puppetmaster_status",
            description="Return task, artifact, and stale lease state for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["status", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_logs",
            description="Return readable Puppetmaster event logs for a job, defaulting to latest.",
            input_schema=job_schema(),
            handler=lambda args: run_cli(["logs"] + optional_job(args), args),
        ),
        McpTool(
            name="puppetmaster_live_artifacts",
            description="Return the live artifact feed for a job without waiting for final stitching.",
            input_schema=feed_schema(),
            handler=lambda args: run_feed(args),
        ),
        McpTool(
            name="puppetmaster_partial_summary",
            description="Return a live summary from current artifacts without waiting for final stitching.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["show", require_string(args, "job_id"), "--partial"], args),
        ),
        McpTool(
            name="puppetmaster_artifacts",
            description="Return structured JSON artifacts for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["artifacts", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_show",
            description="Return the stitched summary for a Puppetmaster job.",
            input_schema=job_schema(required=True),
            handler=lambda args: run_cli(["show", require_string(args, "job_id")], args),
        ),
        McpTool(
            name="puppetmaster_codegraph_search",
            description=(
                "Find symbols by name using the local CodeGraph index. "
                "Bundles `codegraph query` so Cursor Agent only needs the Puppetmaster MCP."
            ),
            input_schema=codegraph_search_schema(),
            handler=run_codegraph_search,
        ),
        McpTool(
            name="puppetmaster_codegraph_context",
            description=(
                "Build task-relevant CodeGraph context (entry points, related symbols) "
                "without spawning a worker. Use for quick repo intel before editing."
            ),
            input_schema=codegraph_context_schema(),
            handler=run_codegraph_context,
        ),
        McpTool(
            name="puppetmaster_codegraph_affected",
            description=(
                "Resolve which test files are impacted by changed source files using "
                "CodeGraph's import graph. Great for targeted CI/test selection."
            ),
            input_schema=codegraph_affected_schema(),
            handler=run_codegraph_affected,
        ),
        McpTool(
            name="puppetmaster_codegraph_files",
            description="Return the indexed file structure from CodeGraph (faster than fs scans).",
            input_schema=codegraph_files_schema(),
            handler=run_codegraph_files,
        ),
        McpTool(
            name="puppetmaster_codegraph_status",
            description="Return CodeGraph index health and statistics for the target workspace.",
            input_schema=base_schema(),
            handler=run_codegraph_status,
        ),
        McpTool(
            name="puppetmaster_codegraph_init",
            description=(
                "Initialize CodeGraph in the target workspace (creates .codegraph/). "
                "Pass index=true to also build the full index immediately."
            ),
            input_schema=codegraph_init_schema(),
            handler=run_codegraph_init,
        ),
    ]


def run_codegraph_search(args: JsonObject) -> JsonObject:
    payload = codegraph_query(
        require_string(args, "query"),
        cwd(args),
        kind=args.get("kind") if isinstance(args.get("kind"), str) else None,
        limit=int(args["limit"]) if args.get("limit") is not None else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_context(args: JsonObject) -> JsonObject:
    payload = codegraph_context_command(
        require_string(args, "task"),
        cwd(args),
        max_nodes=int(args.get("max_nodes") or 15),
        fmt=str(args.get("format") or "markdown"),
    )
    return codegraph_response(payload)


def run_codegraph_affected(args: JsonObject) -> JsonObject:
    files = args.get("files")
    if not isinstance(files, list) or not files:
        return tool_error("files must be a non-empty array of changed source paths.")
    payload = codegraph_affected(
        [str(item) for item in files if str(item).strip()],
        cwd(args),
        depth=int(args["depth"]) if args.get("depth") is not None else None,
        filter_pattern=str(args["filter"]) if args.get("filter") else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_files(args: JsonObject) -> JsonObject:
    payload = codegraph_files_listing(
        cwd(args),
        path=str(args["path"]) if args.get("path") else None,
        fmt=str(args["format"]) if args.get("format") else None,
        filter_pattern=str(args["filter"]) if args.get("filter") else None,
        max_depth=int(args["max_depth"]) if args.get("max_depth") is not None else None,
        json_output=bool(args.get("json", True)),
    )
    return codegraph_response(payload)


def run_codegraph_status(args: JsonObject) -> JsonObject:
    payload = codegraph_status_command(cwd(args))
    return codegraph_response(payload)


def run_codegraph_init(args: JsonObject) -> JsonObject:
    payload = codegraph_init_command(
        cwd(args),
        index=bool(args.get("index", False)),
    )
    return codegraph_response(payload)


def codegraph_response(payload: JsonObject) -> JsonObject:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": not payload.get("ok", False),
    }


def run_cursor(args: JsonObject, review: bool = False, plan: bool = False) -> JsonObject:
    return run_cli(cursor_command(args, review=review, plan=plan), args)


def start_cursor(args: JsonObject, review: bool = False, plan: bool = False) -> JsonObject:
    return start_cli(cursor_command(args, review=review, plan=plan), args)


def cursor_command(args: JsonObject, review: bool = False, plan: bool = False) -> list[str]:
    goal = require_string(args, "goal")
    command = ["cursor", goal, "--cwd", cwd(args), "--dry-run"]
    if review:
        command.append("--review")
    if plan:
        command.append("--plan")
    model = args.get("model")
    if model:
        command.extend(["--model", str(model)])
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds:
        command.extend(["--timeout-seconds", str(timeout_seconds)])
    return command


def run_claude(args: JsonObject) -> JsonObject:
    return run_cli(claude_command(args), args)


def start_claude(args: JsonObject) -> JsonObject:
    return start_cli(claude_command(args), args)


def claude_command(args: JsonObject) -> list[str]:
    goal = require_string(args, "goal")
    command = [
        "claude",
        goal,
        "--cwd",
        cwd(args),
        "--permission-mode",
        str(args.get("permission_mode") or "acceptEdits"),
    ]
    if args.get("model"):
        command.extend(["--model", str(args["model"])])
    if args.get("timeout_seconds"):
        command.extend(["--timeout-seconds", str(args["timeout_seconds"])])
    if args.get("allow_dirty"):
        command.append("--allow-dirty")
    return command


def start_swarm(args: JsonObject) -> JsonObject:
    goal = require_string(args, "goal")
    command = ["run", goal]
    roles = normalized_roles(args)
    adapter = args.get("adapter")
    if args.get("config"):
        command.extend(["--config", str(args["config"])])
    elif adapter:
        config_path = write_generated_swarm_config(args, roles or ["explore"], str(adapter))
        command.extend(["--config", str(config_path)])
    elif roles:
        if not args.get("allow_local_demo"):
            return tool_error(
                "Custom-role MCP swarms require a workflow config or adapter. "
                "Otherwise Puppetmaster would use the demo local adapter and return generic artifacts.",
                {
                    "roles": roles,
                    "fix": "Use puppetmaster_start_cursor_swarm, pass adapter='cursor', pass config, or set allow_local_demo=true for tests/demos.",
                },
            )
        command.append("--workers")
        command.extend(roles)
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    return start_cli(command, args)


def start_cursor_swarm(args: JsonObject) -> JsonObject:
    roles = normalized_roles(args) or [
        "pipeline-mapper",
        "decision-explainer",
        "conflict-auditor",
        "test-coverage-reviewer",
    ]
    config_path = write_generated_swarm_config(args, roles, "cursor")
    command = ["run", require_string(args, "goal"), "--config", str(config_path)]
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    return start_cli(command, args)


def normalized_roles(args: JsonObject) -> list[str]:
    roles = args.get("roles")
    if not isinstance(roles, list):
        return []
    return [str(role) for role in roles if str(role).strip()]


def write_generated_swarm_config(args: JsonObject, roles: list[str], adapter: str) -> Path:
    if adapter not in {"cursor", "local"}:
        raise ValueError(f"MCP swarm adapter is not supported yet: {adapter}")
    root = mcp_state_dir(args)
    config_dir = root / "mcp-configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"swarm_{int(time.time() * 1000)}_{os.getpid()}.json"
    goal = require_string(args, "goal")
    timeout_seconds = int(args.get("timeout_seconds") or 900)
    model = str(args.get("model") or "default")
    workers = []
    for role in roles:
        prompt = (
            f"Role: {role}\n"
            f"Goal: {goal}\n\n"
            "Return structured findings with concrete file/function evidence. "
            "Do not modify files unless the user explicitly requested implementation. "
            "Return only Puppetmaster artifact JSON with an artifacts array."
        )
        payload: JsonObject = {"prompt": prompt, "cwd": cwd(args), "timeout_seconds": timeout_seconds}
        if adapter == "cursor":
            payload["model"] = model
        workers.append(
            {
                "role": role,
                "instruction": prompt,
                "adapter": adapter,
                "payload": payload,
            }
        )
    config_path.write_text(json.dumps({"lease_seconds": 10, "workers": workers}, indent=2), encoding="utf-8")
    return config_path


def run_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(mcp_state_dir(args))
    process = subprocess.run(
        [sys.executable, "-m", "puppetmaster", "--state-dir", state_dir] + command,
        cwd=cwd(args),
        env=launcher_environment(args),
        capture_output=True,
        text=True,
        timeout=int(args.get("runner_timeout_seconds") or 1800),
    )
    body = {
        "command": "python -m puppetmaster " + " ".join(command),
        "cwd": cwd(args),
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(body, indent=2)}],
        "isError": process.returncode != 0,
    }


def run_feed(args: JsonObject) -> JsonObject:
    command = ["feed", require_string(args, "job_id"), "--json"]
    if args.get("limit"):
        command.extend(["--limit", str(args["limit"])])
    return run_cli(command, args)


def start_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(mcp_state_dir(args))
    run_dir = Path(state_dir) / "mcp-runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"mcp_{int(time.time() * 1000)}_{os.getpid()}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
    full_command = [
        sys.executable,
        "-m",
        "puppetmaster",
        "--state-dir",
        state_dir,
        "--emit-job-id-early",
    ] + command
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        full_command,
        cwd=cwd(args),
        env=launcher_environment(args),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    ASYNC_PROCESSES.append(process)
    stdout_handle.close()
    stderr_handle.close()
    job_id = wait_for_job_id(stdout_path, stderr_path, process, timeout_seconds=5)
    body = {
        "run_id": run_id,
        "job_id": job_id,
        "pid": process.pid,
        "command": "python -m puppetmaster " + " ".join(command),
        "cwd": cwd(args),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "next_steps": [
            f"Call puppetmaster_status with job_id={job_id}",
            f"Call puppetmaster_logs with job_id={job_id}",
            f"Call puppetmaster_show with job_id={job_id} after completion",
        ],
    }
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2)}], "isError": False}


def wait_for_job_id(
    stdout_path: Path,
    stderr_path: Path,
    process: subprocess.Popen,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    pattern = re.compile(r"job_id:\s*(job_[A-Za-z0-9]+)")
    while time.monotonic() < deadline:
        if process.poll() is not None and not stdout_path.exists():
            break
        if stdout_path.exists():
            text = stdout_path.read_text(encoding="utf-8")
            match = pattern.search(text)
            if match:
                return match.group(1)
            if process.poll() is not None:
                break
        time.sleep(0.05)
    stderr = stderr_path.read_text(encoding="utf-8")[-1000:] if stderr_path.exists() else ""
    raise RuntimeError(
        f"started Puppetmaster process but did not receive early job_id; "
        f"pid={process.pid}; returncode={process.poll()}; stderr={stderr}"
    )


def launcher_environment(args: JsonObject) -> dict[str, str]:
    env = environment(args)
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else source_root
    )
    return env


def environment(args: JsonObject) -> dict[str, str]:
    env = os.environ.copy()
    if args.get("cursor_api_key"):
        env["CURSOR_API_KEY"] = str(args["cursor_api_key"])
    if args.get("anthropic_api_key"):
        env["ANTHROPIC_API_KEY"] = str(args["anthropic_api_key"])
    if args.get("claude_code_command"):
        env["CLAUDE_CODE_COMMAND"] = str(args["claude_code_command"])
    return env


def cwd(args: JsonObject) -> str:
    return str(args.get("cwd") or os.getcwd())


def mcp_state_dir(args: JsonObject) -> Path:
    value = args.get("state_dir")
    return resolve_state_dir(str(value) if value else None, cwd=Path(cwd(args)))


def require_string(args: JsonObject, name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def optional_job(args: JsonObject) -> list[str]:
    job_id = args.get("job_id")
    return [str(job_id)] if job_id else []


def base_schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "cwd": {"type": "string", "description": "Workspace/repository path."},
            "state_dir": {
                "type": "string",
                "description": (
                    "Optional Puppetmaster state directory, relative to cwd unless absolute. "
                    "Defaults to per-workspace app state outside the repository."
                ),
            },
            "runner_timeout_seconds": {
                "type": "integer",
                "description": "Maximum time to wait for the local Puppetmaster process.",
            },
        },
    }


def job_schema(required: bool = False) -> JsonObject:
    schema = base_schema()
    schema["properties"]["job_id"] = {"type": "string", "description": "Puppetmaster job id."}
    if required:
        schema["required"] = ["job_id"]
    return schema


def feed_schema() -> JsonObject:
    schema = job_schema(required=True)
    schema["properties"]["limit"] = {
        "type": "integer",
        "description": "Limit feed to the most recent N artifacts.",
    }
    return schema


def goal_schema(default_goal: str) -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "goal": {
                "type": "string",
                "description": "Goal/prompt to send to the worker.",
                "default": default_goal,
            },
            "model": {"type": "string", "description": "Optional provider model name."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Worker timeout passed to the adapter.",
            },
            "cursor_api_key": {
                "type": "string",
                "description": "Optional Cursor API key. Prefer MCP env config instead.",
            },
        }
    )
    schema["required"] = ["goal"]
    return schema


def swarm_schema() -> JsonObject:
    schema = goal_schema("Review this repo and produce structured artifacts.")
    schema["properties"].update(
        {
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local worker roles to run.",
            },
            "config": {"type": "string", "description": "Optional workflow config path."},
            "adapter": {
                "type": "string",
                "enum": ["cursor", "local"],
                "description": "Adapter to use for generated role configs. Required for custom roles unless config or allow_local_demo is set.",
            },
            "allow_local_demo": {
                "type": "boolean",
                "default": False,
                "description": "Allow custom roles to use the deterministic local demo adapter.",
            },
            "worker_mode": {
                "type": "string",
                "enum": ["subprocess", "inline", "daemon"],
                "default": "subprocess",
                "description": "Worker execution mode.",
            },
        }
    )
    return schema


def cursor_swarm_schema() -> JsonObject:
    schema = swarm_schema()
    schema["properties"].pop("adapter", None)
    schema["properties"].pop("allow_local_demo", None)
    schema["properties"]["worker_mode"]["default"] = "subprocess"
    return schema


def codegraph_search_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "query": {
                "type": "string",
                "description": "Symbol name or substring to search the local CodeGraph index for.",
            },
            "kind": {
                "type": "string",
                "description": "Optional symbol kind filter (e.g. function, class, method, route).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches to return.",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    schema["required"] = ["query"]
    return schema


def codegraph_context_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "task": {
                "type": "string",
                "description": "Natural-language task description; CodeGraph returns relevant entry points and symbols.",
            },
            "max_nodes": {
                "type": "integer",
                "default": 15,
                "description": "Upper bound on graph nodes returned in the context bundle.",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "json", "text"],
                "default": "markdown",
                "description": "Output format for the context bundle.",
            },
        }
    )
    schema["required"] = ["task"]
    return schema


def codegraph_affected_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Changed source file paths (relative to cwd) whose tests should be discovered.",
            },
            "depth": {
                "type": "integer",
                "description": "Max dependency traversal depth (CodeGraph default: 5).",
            },
            "filter": {
                "type": "string",
                "description": "Custom glob used to identify test files (e.g. 'tests/**/*.py').",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    schema["required"] = ["files"]
    return schema


def codegraph_files_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"].update(
        {
            "path": {
                "type": "string",
                "description": "Optional sub-path to scope the file listing.",
            },
            "format": {
                "type": "string",
                "description": "CodeGraph format option for the listing (e.g. tree, list).",
            },
            "filter": {
                "type": "string",
                "description": "Glob filter applied to the listing.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth to display.",
            },
            "json": {
                "type": "boolean",
                "default": True,
                "description": "Return CodeGraph output as JSON. Set false for human-readable text.",
            },
        }
    )
    return schema


def codegraph_init_schema() -> JsonObject:
    schema = base_schema()
    schema["properties"]["index"] = {
        "type": "boolean",
        "default": False,
        "description": "If true, run a full index immediately after initialization.",
    }
    return schema


def claude_schema() -> JsonObject:
    schema = goal_schema("Implement the requested change and run focused tests.")
    schema["properties"].update(
        {
            "permission_mode": {
                "type": "string",
                "default": "acceptEdits",
                "description": "Claude Code permission mode.",
            },
            "allow_dirty": {
                "type": "boolean",
                "default": False,
                "description": "Allow Claude Code to run in a dirty working tree.",
            },
            "anthropic_api_key": {
                "type": "string",
                "description": "Optional Anthropic API key. Prefer MCP env config instead.",
            },
            "claude_code_command": {
                "type": "string",
                "description": "Optional Claude Code command, such as npx -y @anthropic-ai/claude-code.",
            },
        }
    )
    return schema


def tool_to_json(tool: McpTool) -> JsonObject:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def tool_error(message: str, payload: Optional[JsonObject] = None) -> JsonObject:
    body: JsonObject = {"error": message}
    if payload:
        body.update(payload)
    return {"content": [{"type": "text", "text": json.dumps(body, indent=2)}], "isError": True}


def error_response(request_id: Any, code: int, message: str) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
