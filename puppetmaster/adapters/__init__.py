"""Worker adapters — subprocess runners and artifact emitters for Puppetmaster.

This package is the facade for ``puppetmaster.adapters``; all public names from
the former monolithic module are re-exported here unchanged.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union

from puppetmaster.codegraph import (
    enrich_prompt_with_codegraph,
    inject_worker_cli_env,
    repo_file_census,
    scrub_foreign_interpreter_env,
)
from puppetmaster.failure import (
    NOT_AUTHENTICATED,
    classify_claude_code_failure,
    classify_codex_failure,
    classify_cursor_failure,
    classify_hermes_failure,
    classify_openai_failure,
)
from puppetmaster.fs_permissions import mkdir_private, open_private, write_private_text
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.openai_security import (
    DEFAULT_OPENAI_BASE_URL,
    validate_openai_base_url_for_task,
)
from puppetmaster.redaction import redact_secrets
from puppetmaster.usage import token_usage

from ._base import (
    AdapterInfo,
    CliInvocation,
    CliWorkerAdapter,
    FullEditWorkerAdapter,
    WorkerAdapter,
    build_patch_payload,
    command_parts,
    diff_source_payload,
    dirty_worktree_guard,
    dirty_worktree_paths_note,
    failure_verification,
    make_patch_artifact,
    missing_cli_artifact,
    resolve_command,
    snapshot_has_diff,
    tool_list,
    verification_artifact,
)
from puppetmaster.ports import apply_worktree_ports

from ._git import (
    GitSnapshot,
    git_diff_output,
    git_lines,
    git_output,
    git_snapshot,
    git_untracked_diff,
    git_untracked_files,
    git_worktree_root,
    git_worktree_tree,
    worktree_guard,
)
from ._prompts import (
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    prompt_with_skills,
    with_job_brief,
    with_repo_census,
    with_report_contract,
)
from ._streaming import (
    StreamedProcess,
    capture_subprocess_stdout,
    run_streamed_subprocess,
)
from .claude_code import (
    DEFAULT_CLAUDE_CODE_MODEL,
    ClaudeCodeAdapter,
    build_claude_code_command,
    is_bedrock_model_id,
    resolve_claude_code_model,
)
from .codex import (
    DEFAULT_CODEX_MODEL,
    CodexAdapter,
    build_codex_exec_command,
    last_codex_agent_message,
    parse_codex_events,
)
from .cursor import (
    CursorAdapter,
    cursor_artifact_from_item,
    cursor_degraded_artifact,
    cursor_result_artifacts,
    cursor_result_text,
    implement_report_artifacts,
    parse_cursor_artifact_payload,
    sdk_usage_from_stdout,
)
from .hermes import (
    DEFAULT_HERMES_ANALYZE_TOOLSETS,
    DEFAULT_HERMES_IMPLEMENT_TOOLSETS,
    VALID_HERMES_REASONING_EFFORTS,
    HermesAdapter,
    available_hermes_providers,
    build_hermes_chat_command,
    hermes_credentials_available,
    hermes_reasoning_effort_env,
    prune_hermes_tool_sessions,
)
from .local import LocalAdapter, ShellAdapter, UnconfiguredProviderAdapter
from .openai import DEFAULT_OPENAI_MODEL, OpenAIAdapter
from .registry import ADAPTER_INFO, ADAPTERS, get_adapter

# Internal helpers imported by tests and sibling modules
from ._base import _should_emit_patch_artifact
from ._prompts import (
    _ANALYZE_JSON_ONLY_RETRY,
    _IMPLEMENT_REPORT_CONTRACT,
    _MEMORY_MAX_ITEMS,
    _MEMORY_STATEMENT_MAX_CHARS,
)
from .hermes import _hermes_session_cleanup_enabled

__all__ = [
    "ADAPTERS",
    "ADAPTER_INFO",
    "AdapterInfo",
    "Artifact",
    "ArtifactType",
    "ClaudeCodeAdapter",
    "CliInvocation",
    "CliWorkerAdapter",
    "CodexAdapter",
    "CursorAdapter",
    "DEFAULT_CLAUDE_CODE_MODEL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_HERMES_ANALYZE_TOOLSETS",
    "DEFAULT_HERMES_IMPLEMENT_TOOLSETS",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "FullEditWorkerAdapter",
    "GitSnapshot",
    "HermesAdapter",
    "LocalAdapter",
    "NOT_AUTHENTICATED",
    "OpenAIAdapter",
    "ShellAdapter",
    "StreamedProcess",
    "Task",
    "UnconfiguredProviderAdapter",
    "VALID_HERMES_REASONING_EFFORTS",
    "WorkerAdapter",
    "apply_worktree_ports",
    "available_hermes_providers",
    "build_claude_code_command",
    "build_codex_exec_command",
    "build_hermes_chat_command",
    "build_implement_prompt",
    "build_patch_payload",
    "build_structured_prompt",
    "capture_subprocess_stdout",
    "classify_claude_code_failure",
    "classify_codex_failure",
    "classify_cursor_failure",
    "classify_hermes_failure",
    "classify_openai_failure",
    "command_parts",
    "cursor_artifact_from_item",
    "cursor_degraded_artifact",
    "cursor_result_artifacts",
    "cursor_result_text",
    "diff_source_payload",
    "dirty_worktree_guard",
    "dirty_worktree_paths_note",
    "enrich_prompt_with_codegraph",
    "failure_verification",
    "get_adapter",
    "git_diff_output",
    "git_lines",
    "git_output",
    "git_snapshot",
    "git_untracked_diff",
    "git_untracked_files",
    "git_worktree_root",
    "git_worktree_tree",
    "hermes_credentials_available",
    "hermes_reasoning_effort_env",
    "implement_report_artifacts",
    "inject_worker_cli_env",
    "is_bedrock_model_id",
    "last_codex_agent_message",
    "make_patch_artifact",
    "missing_cli_artifact",
    "mkdir_private",
    "open_private",
    "parse_codex_events",
    "parse_cursor_artifact_payload",
    "prompt_with_memory",
    "prompt_with_skills",
    "prune_hermes_tool_sessions",
    "redact_secrets",
    "repo_file_census",
    "resolve_claude_code_model",
    "resolve_command",
    "run_streamed_subprocess",
    "scrub_foreign_interpreter_env",
    "sdk_usage_from_stdout",
    "snapshot_has_diff",
    "token_usage",
    "tool_list",
    "validate_openai_base_url_for_task",
    "verification_artifact",
    "with_job_brief",
    "with_repo_census",
    "with_report_contract",
    "worktree_guard",
    "write_private_text",
]
