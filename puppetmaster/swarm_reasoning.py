"""One swarm reasoning default for every adapter.

Callers pin effort on the task payload. Catalog High+Fast / payload_defaults
are not pins. ``merge_routing_payload`` stamps the default and overlays each
adapter dialect. GPT-5.6 Completions ``none`` stays a wire constraint in
``providers._openai_api_chat``, not a pin.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

DEFAULT_SWARM_REASONING_EFFORT = "medium"


def _normalize_effort(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _effort_from_extra_args(extra_args: object) -> Optional[str]:
    if not isinstance(extra_args, (list, tuple)):
        return None
    tokens = [str(part) for part in extra_args]
    for index, token in enumerate(tokens):
        if token == "--effort" and index + 1 < len(tokens):
            return _normalize_effort(tokens[index + 1])
        if token.startswith("--effort="):
            return _normalize_effort(token.split("=", 1)[1])
        if token.startswith("model_reasoning_effort="):
            return _normalize_effort(token.split("=", 1)[1])
        if token == "-c" and index + 1 < len(tokens):
            nxt = tokens[index + 1]
            if nxt.startswith("model_reasoning_effort="):
                return _normalize_effort(nxt.split("=", 1)[1])
    return None


def _effort_from_params(params: object) -> Optional[str]:
    if not isinstance(params, list):
        return None
    for item in params:
        if isinstance(item, dict) and str(item.get("id") or "") == "effort":
            return _normalize_effort(item.get("value"))
    return None


def caller_pinned_effort(payload: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return a caller-only pin. Catalog defaults are not pins."""
    if not payload:
        return None
    for key in ("reasoning_effort", "effort"):
        pinned = _normalize_effort(payload.get(key))
        if pinned:
            return pinned
    pinned = _effort_from_extra_args(payload.get("extra_args"))
    if pinned:
        return pinned
    return _effort_from_params(payload.get("params"))


def _replace_flag_pair(args: list[Any], flag: str, value: str) -> tuple[list[Any], bool]:
    rewritten: list[Any] = []
    replaced = False
    index = 0
    while index < len(args):
        token = str(args[index])
        if token == flag and index + 1 < len(args):
            rewritten.extend([flag, value])
            index += 2
            replaced = True
            continue
        if token.startswith(flag + "="):
            rewritten.append(flag + "=" + value)
            index += 1
            replaced = True
            continue
        rewritten.append(args[index])
        index += 1
    return rewritten, replaced


def _replace_codex_effort(args: list[Any], value: str) -> tuple[list[Any], bool]:
    rewritten: list[Any] = []
    replaced = False
    index = 0
    while index < len(args):
        token = str(args[index])
        nxt = str(args[index + 1]) if index + 1 < len(args) else ""
        if token == "-c" and nxt.startswith("model_reasoning_effort="):
            rewritten.extend(["-c", "model_reasoning_effort=" + value])
            index += 2
            replaced = True
            continue
        if token.startswith("model_reasoning_effort="):
            rewritten.append("model_reasoning_effort=" + value)
            index += 1
            replaced = True
            continue
        rewritten.append(args[index])
        index += 1
    return rewritten, replaced


def _overlay_params(payload: dict[str, Any], effort: str, adapter: Optional[str]) -> None:
    params = payload.get("params")
    if isinstance(params, list):
        rewritten = []
        found = False
        for item in params:
            if isinstance(item, dict) and str(item.get("id") or "") == "effort":
                entry = dict(item)
                entry["value"] = effort
                rewritten.append(entry)
                found = True
            else:
                rewritten.append(item)
        if not found and adapter == "cursor":
            rewritten.append({"id": "effort", "value": effort})
        payload["params"] = rewritten
        return
    if adapter == "cursor":
        payload["params"] = [{"id": "effort", "value": effort}]


def _overlay_extra_args(payload: dict[str, Any], effort: str, adapter: Optional[str]) -> None:
    raw = payload.get("extra_args")
    args = list(raw) if isinstance(raw, (list, tuple)) else []
    args, replaced_claude = _replace_flag_pair(args, "--effort", effort)
    args, replaced_codex = _replace_codex_effort(args, effort)
    if adapter == "claude-code" and not replaced_claude:
        args.extend(["--effort", effort])
    if adapter == "codex" and not replaced_codex:
        args.extend(["-c", "model_reasoning_effort=" + effort])
    if args:
        payload["extra_args"] = args


def _slug_encodes_effort(model: object) -> bool:
    if not model:
        return False
    return str(model).lower().endswith(("-high", "-medium", "-low"))


def _overlay_antigravity(payload: dict[str, Any], effort: str, adapter: Optional[str]) -> None:
    if adapter not in (None, "antigravity"):
        return
    if adapter != "antigravity" and "effort" not in payload:
        return
    if _slug_encodes_effort(payload.get("model")):
        return
    payload["effort"] = effort


def overlay_adapter_dialect(
    payload: dict[str, Any],
    effort: str,
    adapter: Optional[str] = None,
) -> dict[str, Any]:
    """Map ``reasoning_effort`` onto adapter knobs in place."""
    payload["reasoning_effort"] = effort
    _overlay_params(payload, effort, adapter)
    _overlay_extra_args(payload, effort, adapter)
    _overlay_antigravity(payload, effort, adapter)
    return payload


def apply_swarm_reasoning(
    merged: dict[str, Any],
    caller_payload: Optional[Mapping[str, Any]],
    adapter: Optional[str] = None,
) -> dict[str, Any]:
    """Stamp medium unless the caller pinned; overlay adapter dialect."""
    pin = caller_pinned_effort(caller_payload)
    effort = pin or DEFAULT_SWARM_REASONING_EFFORT
    return overlay_adapter_dialect(merged, effort, adapter)
