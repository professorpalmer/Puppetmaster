"""Durable model-registry authority for routed and explicitly pinned tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from puppetmaster.model_registry import (
    ModelSpec,
    default_registry_path,
    load_registry,
    registry_digest,
    resolve_model_pin,
    stamp_resolved_model_pin,
)


class RegistryAuthorityError(RuntimeError):
    """Raised when persisted model authority is missing, stale, or unusable."""


def registry_path_from_payload(
    payload: dict,
    *,
    allow_default: bool,
) -> Path:
    raw = (payload or {}).get("registry_path")
    if raw is None or not str(raw).strip():
        if not allow_default:
            raise RegistryAuthorityError(
                "task is missing durable registry_path authority"
            )
        return default_registry_path().expanduser().resolve()
    return Path(str(raw)).expanduser().resolve()


def bind_registry_authority(
    payload: dict,
    registry_path: Path,
    registry: Iterable[ModelSpec],
) -> dict:
    """Return a payload bound to an exact registry path and content epoch."""
    materialized = list(registry)
    return {
        **(payload or {}),
        "registry_path": str(Path(registry_path).expanduser().resolve()),
        "registry_digest": registry_digest(materialized),
    }


def load_bound_registry(payload: dict) -> tuple[Path, list[ModelSpec], str]:
    """Load task authority and fail closed when its digest has drifted."""
    path = registry_path_from_payload(payload, allow_default=False)
    expected = str((payload or {}).get("registry_digest") or "").strip()
    if not expected:
        raise RegistryAuthorityError(
            f"task registry authority at {path} is missing registry_digest"
        )
    registry = load_registry(path)
    actual = registry_digest(registry)
    if actual != expected:
        raise RegistryAuthorityError(
            "registry authority drift: digest mismatch for "
            f"{path} (expected {expected}, found {actual})"
        )
    return path, registry, actual


def resolve_and_bind_explicit_pin(
    payload: dict,
    *,
    adapter: str,
    registry_path: Optional[Path] = None,
) -> dict:
    """Resolve one live explicit pin and bind the registry authority used."""
    source = str(
        (payload or {}).get("pinned_model")
        or (payload or {}).get("model")
        or ""
    ).strip()
    if not source:
        return dict(payload or {})
    path = registry_path or registry_path_from_payload(payload, allow_default=True)
    registry = load_registry(path)
    pin = resolve_model_pin(source, registry, adapter=adapter)
    if pin is None:
        raise RegistryAuthorityError(
            f"model pin {source!r} for adapter {adapter!r} is not enabled and "
            f"routable in registry {path}; it may be disabled, retired, or absent"
        )
    stamped = stamp_resolved_model_pin(dict(payload or {}), pin)
    return bind_registry_authority(stamped, path, registry)


def validate_pinned_dispatch(payload: dict, *, adapter: str) -> dict:
    """Canonicalize/revalidate explicit pin authority before worker dispatch.

    Current tasks carry ``pinned_model``. Legacy persisted tasks may carry only
    ``model`` with ``auto_route=false``; those are still explicit pins and are
    upgraded to the durable stamped representation before invocation.
    """
    pinned = str((payload or {}).get("pinned_model") or "").strip()
    if not pinned:
        legacy_model = str((payload or {}).get("model") or "").strip()
        if not legacy_model or bool((payload or {}).get("auto_route")):
            return dict(payload or {})
        if (payload or {}).get("registry_path") and (payload or {}).get(
            "registry_digest"
        ):
            path, registry, _digest = load_bound_registry(payload)
            pin = resolve_model_pin(legacy_model, registry, adapter=adapter)
            if pin is None:
                raise RegistryAuthorityError(
                    f"legacy model pin {legacy_model!r} is disabled, retired, "
                    f"or absent in registry {path}"
                )
            return bind_registry_authority(
                stamp_resolved_model_pin(dict(payload or {}), pin),
                path,
                registry,
            )
        return resolve_and_bind_explicit_pin(payload, adapter=adapter)

    path, registry, _digest = load_bound_registry(payload)
    pin = resolve_model_pin(pinned, registry, adapter=adapter)
    if pin is None:
        raise RegistryAuthorityError(
            f"model pin {pinned!r} is disabled, retired, or absent in registry {path}"
        )
    expected_wire = str((payload or {}).get("pinned_adapter_model_name") or "")
    executable_wire = str((payload or {}).get("model") or "")
    routed_id = str((payload or {}).get("router_model_id") or "")
    if (
        not expected_wire
        or executable_wire != expected_wire
        or pin.adapter_model_name != expected_wire
    ):
        raise RegistryAuthorityError(
            f"registry authority invalid for model pin {pinned!r}: executable "
            "model and pinned wire identity diverge"
        )
    if routed_id != pinned or pin.registry_id != pinned:
        raise RegistryAuthorityError(
            f"registry authority invalid for model pin {pinned!r}: "
            "router_model_id and pinned registry identity diverge"
        )
    return dict(payload or {})
