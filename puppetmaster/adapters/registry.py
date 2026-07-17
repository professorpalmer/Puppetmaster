from __future__ import annotations

from typing import Optional

from ._base import AdapterInfo, WorkerAdapter
from .agentic import AgenticAdapter
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .cursor import CursorAdapter
from .hermes import HermesAdapter
from .local import LocalAdapter, ShellAdapter
from .openai import OpenAIAdapter

ADAPTERS: dict[str, WorkerAdapter] = {
    "local": LocalAdapter(),
    "shell": ShellAdapter(),
    "agentic": AgenticAdapter(),
    "cursor": CursorAdapter(),
    "claude-code": ClaudeCodeAdapter(),
    "openai": OpenAIAdapter(),
    "codex": CodexAdapter(),
    "hermes": HermesAdapter(),
}


ADAPTER_INFO = [
    AdapterInfo(
        name="local",
        status="built-in",
        description="Deterministic structured artifacts for demo/runtime roles.",
        requires=[],
    ),
    AdapterInfo(
        name="agentic",
        status="built-in",
        description=(
            "Standalone provider-agnostic worker: runs its own tool-use loop "
            "against a provider API directly (OpenAI-compatible, Anthropic, or "
            "AWS Bedrock Converse) for analyze and full-edit implement modes. "
            "No external agent CLI required — provider keys or AWS IAM auth."
        ),
        requires=[
            "a provider API key (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "GEMINI_API_KEY, OPENROUTER_API_KEY) or AWS Bedrock credentials "
            "(AWS_ACCESS_KEY_ID / ~/.aws / AWS_BEARER_TOKEN_BEDROCK)"
        ],
    ),
    AdapterInfo(
        name="shell",
        status="built-in",
        description="Runs bounded shell commands and emits verification artifacts.",
        requires=[],
    ),
    AdapterInfo(
        name="cursor",
        status="optional",
        description="Runs local Cursor SDK one-shot agents.",
        requires=["node", "npm install", "CURSOR_API_KEY"],
    ),
    AdapterInfo(
        name="claude-code",
        status="optional",
        description="Runs the Claude Code CLI in non-interactive full-edit mode.",
        requires=["claude CLI", "Claude Code auth"],
    ),
    AdapterInfo(
        name="openai",
        status="optional",
        description="Calls the OpenAI Chat Completions API directly with OPENAI_API_KEY.",
        requires=["OPENAI_API_KEY"],
    ),
    AdapterInfo(
        name="codex",
        status="optional",
        description=(
            "Runs the official OpenAI Codex CLI (`codex exec --json`) "
            "non-interactively. Captures billing-grade token counts from the "
            "structured event stream and emits Puppetmaster artifacts plus a "
            "PATCH artifact when the agent edits files."
        ),
        requires=[
            "codex CLI (`npm install -g @openai/codex`)",
            "OPENAI_API_KEY or `codex login`",
        ],
    ),
    AdapterInfo(
        name="hermes",
        status="optional",
        description=(
            "Runs the NousResearch Hermes CLI (`hermes chat`) headlessly for "
            "analyze and full-edit implement modes. Launches in an isolated "
            "process session and attributes file edits via git diff rather "
            "than exit code."
        ),
        requires=[
            "hermes CLI on PATH",
            "provider credential in ~/.hermes/.env or `hermes login` OAuth",
        ],
    ),
]


def get_adapter(name: str) -> WorkerAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"unsupported adapter: {name}")
    return ADAPTERS[name]


def adapter_runtime_capabilities(name: str) -> dict[str, Optional[str]]:
    """Expose conservative runtime/catalog capabilities for diagnostics.

    Isolation is opt-in and descriptive: the registry must never infer that
    one adapter's state workaround is safe for another harness.
    """
    adapter = get_adapter(name)
    return {
        "state_isolation": getattr(adapter, "state_isolation", "none"),
        "catalog_source": getattr(adapter, "catalog_source", None),
    }
