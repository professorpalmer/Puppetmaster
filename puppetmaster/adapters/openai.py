from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.failure import NOT_AUTHENTICATED, classify_openai_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.openai_security import (
    DEFAULT_OPENAI_BASE_URL,
    validate_openai_base_url_for_task,
)
from puppetmaster.redaction import redact_secrets

from ._base import verification_artifact
from ._facade import facade
from ._prompts import (
    build_structured_prompt,
    prompt_with_memory,
    with_repo_census,
)
from ._streaming import (
    _STDOUT_HEAD_CHARS,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
)
from .cursor import cursor_result_artifacts

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAIAdapter:
    """Calls the OpenAI Chat Completions API directly via OPENAI_API_KEY.

    Mirrors CursorAdapter semantics: enrich the prompt with CodeGraph + promoted
    memory, demand a strict JSON artifact contract, then parse and emit the same
    finding/risk/decision artifacts that the rest of Puppetmaster reasons over.
    """

    name = "openai"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model") or DEFAULT_OPENAI_MODEL
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(build_structured_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = facade("with_repo_census")(prompt, cwd)

        api_key = task.payload.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = (
            task.payload.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        organization = task.payload.get("openai_organization") or os.environ.get(
            "OPENAI_ORG_ID"
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 300))
        evidence_base = [
            "adapter:openai",
            f"model:{model}",
        ] + (["context:codegraph"] if codegraph_used else [])

        if not api_key:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.5,
                    evidence=evidence_base + [NOT_AUTHENTICATED],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": NOT_AUTHENTICATED,
                        "stderr": (
                            "OPENAI_API_KEY is not set. Export it or pass openai_api_key "
                            "in the task payload."
                        ),
                    },
                )
            ]

        base_url_error = validate_openai_base_url_for_task(base_url, task)
        if base_url_error is not None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.5,
                    evidence=evidence_base + ["untrusted_base_url"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "untrusted_base_url",
                        "stderr": base_url_error,
                    },
                )
            ]

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        # Optional knobs — only included when the caller explicitly opts in.
        # OpenAI's GPT-5+ family deprecated `max_tokens` in favor of
        # `max_completion_tokens`. We send the new name by default and let
        # the caller force the legacy name for OpenAI-compatible providers
        # that still require it.
        max_completion = task.payload.get(
            "max_output_tokens",
            task.payload.get("max_completion_tokens", task.payload.get("max_tokens")),
        )
        if max_completion is not None:
            if task.payload.get("legacy_max_tokens"):
                body["max_tokens"] = int(max_completion)
            else:
                body["max_completion_tokens"] = int(max_completion)
        if "temperature" in task.payload:
            body["temperature"] = float(task.payload["temperature"])
        if task.payload.get("reasoning_effort"):
            body["reasoning_effort"] = str(task.payload["reasoning_effort"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "puppetmaster-openai-adapter",
        }
        if organization:
            headers["OpenAI-Organization"] = str(organization)

        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # 4xx/5xx
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + [f"http_status:{exc.code}"],
                    payload={
                        "returncode": exc.code,
                        "model": model,
                        "failure": classify_openai_failure(err_body, exc.code),
                        "stderr": _redacted_tail(err_body, 8000),
                    },
                )
            ]
        except (socket.timeout, TimeoutError):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["timeout"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "timeout",
                        "stderr": f"OpenAI request exceeded {timeout_seconds}s",
                    },
                )
            ]
        except urllib.error.URLError as exc:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["network_error"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "network_error",
                        "stderr": str(exc),
                    },
                )
            ]

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["malformed_response"],
                    payload={
                        "returncode": status_code,
                        "model": model,
                        "failure": "malformed_response",
                        "stderr": _redacted_tail(raw_body, 8000),
                    },
                )
            ]

        choices = data.get("choices") or []
        message = (
            choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        )
        result_text = str(message.get("content") or "").strip()
        finish_reason = (
            choices[0].get("finish_reason") if isinstance(choices, list) and choices else None
        )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        parsed_artifacts = cursor_result_artifacts(task, worker_id, result_text, adapter="openai")
        degraded = not parsed_artifacts and bool(result_text)
        failed = finish_reason not in (None, "stop", "length") and not parsed_artifacts

        stdout_capture = capture_subprocess_stdout(
            text=result_text,
            task=task,
            sidecar_name="openai_response",
        )

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="openai",
            check=task.instruction,
            result=(
                "failed"
                if failed
                else "degraded"
                if degraded
                else "passed"
            ),
            confidence=0.55 if failed else 0.65 if degraded else 0.9,
            evidence=evidence_base + [f"finish:{finish_reason}"] if finish_reason else evidence_base,
            payload={
                "returncode": status_code,
                "model": model,
                "finish_reason": finish_reason,
                "stdout": result_text[-_STDOUT_TAIL_CHARS:],
                "stderr": "",
                "stdout_capture": stdout_capture,
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
                "tokens_total": total_tokens,
                "failure": (
                    "empty_or_unstructured_openai_result"
                    if degraded
                    else None
                    if not failed
                    else "unexpected_finish_reason"
                ),
            },
        )
        artifacts: list[Artifact] = [verification]
        if degraded:
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.RISK,
                    created_by=worker_id,
                    confidence=0.85,
                    evidence=["adapter:openai", "result:empty-or-unstructured"],
                    payload={
                        "risk": "OpenAI call completed without structured Puppetmaster findings.",
                        "mitigation": (
                            "Treat this swarm as degraded; rerun with a stricter prompt or "
                            "inspect the repo directly before implementation."
                        ),
                        "stdout_excerpt": result_text[:_STDOUT_HEAD_CHARS],
                        "stdout_capture": stdout_capture,
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts

