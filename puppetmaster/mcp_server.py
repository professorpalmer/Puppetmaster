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
    ]


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
    roles = args.get("roles")
    if isinstance(roles, list) and roles:
        command.append("--workers")
        command.extend(str(role) for role in roles)
    if args.get("config"):
        command.extend(["--config", str(args["config"])])
    worker_mode = args.get("worker_mode")
    if worker_mode:
        command.extend(["--worker-mode", str(worker_mode)])
    return start_cli(command, args)


def run_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(args.get("state_dir") or ".puppetmaster")
    process = subprocess.run(
        [sys.executable, "-m", "puppetmaster", "--state-dir", state_dir] + command,
        cwd=cwd(args),
        env=environment(args),
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


def start_cli(command: list[str], args: JsonObject) -> JsonObject:
    state_dir = str(args.get("state_dir") or ".puppetmaster")
    run_dir = Path(cwd(args)) / state_dir / "mcp-runs"
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
                "description": "Puppetmaster state directory, relative to cwd unless absolute.",
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
            "worker_mode": {
                "type": "string",
                "enum": ["subprocess", "inline", "daemon"],
                "default": "subprocess",
                "description": "Worker execution mode.",
            },
        }
    )
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


def error_response(request_id: Any, code: int, message: str) -> JsonObject:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
