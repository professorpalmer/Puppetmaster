"""User-owned registry of LLM models for the Puppetmaster router.

This is intentionally a **declarative, user-managed** registry instead
of a hardcoded list of model names + prices. Model availability and
pricing change constantly, every user has different keys and rate
limits, and capability is subjective. The right shape: the user
describes their own models, the router uses what they describe.

The registry lives at ``~/.puppetmaster/models.json`` by default
(overridable with ``PUPPETMASTER_MODELS_PATH`` or ``--registry-path``).
It is **not** committed to the repo because it can contain pricing
preferences and references to environment-variable names that resolve
to API keys.

Each entry pairs:

* a stable user-chosen ``id`` (e.g. ``anthropic/claude-opus``)
* the Puppetmaster ``adapter`` that knows how to invoke it
  (``claude-code``, ``cursor``, or a future raw HTTP adapter)
* ``adapter_model_name``: the literal string the adapter passes
  through to the underlying SDK / CLI / API
* user-asserted ``capability_score`` (0–100)
* ``input_per_mtok_usd`` / ``output_per_mtok_usd`` for cost estimation
* ``context_window`` in tokens
* free-form ``tags`` for policy matching (e.g. ``frontier``, ``cheap``,
  ``long-context``, ``reasoning``, ``code``)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


REGISTRY_ENV = "PUPPETMASTER_MODELS_PATH"


@dataclass(frozen=True)
class ModelSpec:
    """One model in the registry.

    See module docstring for field semantics. ``id`` is user-chosen and
    must be unique within the registry. ``adapter`` must be the name
    of a Puppetmaster adapter that exists at runtime — bad names will
    surface as routing failures, not silent fallbacks.
    """

    id: str
    adapter: str
    adapter_model_name: str
    capability_score: int = 50
    input_per_mtok_usd: float = 0.0
    output_per_mtok_usd: float = 0.0
    context_window: int = 0
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.capability_score <= 100:
            raise ValueError(
                f"capability_score for {self.id} must be 0..100, got {self.capability_score}"
            )
        if self.input_per_mtok_usd < 0 or self.output_per_mtok_usd < 0:
            raise ValueError(f"per-token cost for {self.id} must be non-negative")

    def estimate_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """USD cost estimate for one call. Linear; ignores caching / batching."""
        return (
            (tokens_in / 1_000_000.0) * self.input_per_mtok_usd
            + (tokens_out / 1_000_000.0) * self.output_per_mtok_usd
        )


def default_registry_path() -> Path:
    """Where the registry lives by default.

    Honors ``$PUPPETMASTER_MODELS_PATH`` first; otherwise uses
    ``~/.puppetmaster/models.json``. We deliberately keep this OUT of
    the per-project state dir because models are a per-user concept.
    """
    env = os.environ.get(REGISTRY_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".puppetmaster" / "models.json"


def load_registry(path: Optional[Path] = None) -> list[ModelSpec]:
    """Read the registry from disk. Returns [] if no file exists.

    Missing fields fall back to dataclass defaults so users can author
    minimal entries. Unknown fields are tolerated and dropped (forward
    compat for future Puppetmaster releases).
    """
    resolved = path or default_registry_path()
    if not resolved.is_file():
        return []
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"models registry at {resolved} is not valid JSON: {exc}") from exc
    entries = raw.get("models", raw if isinstance(raw, list) else [])
    specs: list[ModelSpec] = []
    for entry in entries:
        kwargs = {k: v for k, v in entry.items() if k in ModelSpec.__dataclass_fields__}
        specs.append(ModelSpec(**kwargs))
    return specs


def save_registry(
    specs: Iterable[ModelSpec], path: Optional[Path] = None
) -> Path:
    """Persist the registry. Creates parent dirs. Returns the written path."""
    resolved = path or default_registry_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "models": [_spec_to_jsonable(spec) for spec in specs],
    }
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return resolved


def _spec_to_jsonable(spec: ModelSpec) -> dict[str, Any]:
    data = asdict(spec)
    # Drop fields equal to their default to keep the file readable.
    for k, default_value in [
        ("enabled", True),
        ("notes", ""),
        ("tags", []),
        ("context_window", 0),
    ]:
        if data.get(k) == default_value:
            data.pop(k)
    return data


def starter_registry() -> list[ModelSpec]:
    """A starter registry organized into the four tiers most users
    actually think in: fast/cheap, balanced, high-quality, frontier.

    Tier IDs (``cursor/composer-2-5``, ``cursor/gpt-5-5``,
    ``claude-code/opus-4-6``, ``claude-code/opus-4-7``) reflect a
    common mental model where the cheap tier is Cursor's house model,
    the balanced tier is GPT, and the frontier tiers are Anthropic
    Opus. **The ``adapter_model_name`` values default to model strings
    that work *today***; the IDs use the labels you'll eventually want
    so you only have to edit ``adapter_model_name`` when newer
    versions land. Each entry's ``notes`` field flags what to edit.

    Adapter coverage is real (claude-code and cursor adapters both
    exist and accept a ``model`` argument). Capability scores and
    prices are user-asserted starting points — edit them to match
    your subscriptions.
    """
    return [
        ModelSpec(
            id="cursor/composer-2-5",
            adapter="cursor",
            adapter_model_name="composer-1",
            capability_score=55,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
            context_window=0,
            tags=["cursor", "cheap", "fast", "reading", "code"],
            notes=(
                "Fast/cheap tier. Use for high-throughput reading and "
                "low-stakes tasks. PLACEHOLDER: edit adapter_model_name "
                "to 'composer-2-5' (or whatever Cursor SDK accepts) when "
                "Composer 2.5 is released."
            ),
        ),
        ModelSpec(
            id="cursor/gpt-5-5",
            adapter="cursor",
            adapter_model_name="gpt-5",
            capability_score=78,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
            context_window=0,
            tags=["cursor", "balanced", "fast", "vision"],
            notes=(
                "Balanced tier. Medium cost, medium speed, medium "
                "capability. PLACEHOLDER: edit adapter_model_name to "
                "'gpt-5-5' when available. Pricing is $0 because cost "
                "rolls into your Cursor plan."
            ),
        ),
        ModelSpec(
            id="claude-code/opus-4-6",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-5",
            capability_score=88,
            input_per_mtok_usd=15.0,
            output_per_mtok_usd=75.0,
            context_window=200_000,
            tags=["claude", "quality", "vision", "code", "reasoning"],
            notes=(
                "High-quality tier. Medium-high cost, medium speed. "
                "PLACEHOLDER: edit adapter_model_name to 'claude-opus-4-6' "
                "(or the exact Anthropic id) when Opus 4.6 is released."
            ),
        ),
        ModelSpec(
            id="claude-code/opus-4-7",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-5",
            capability_score=96,
            input_per_mtok_usd=15.0,
            output_per_mtok_usd=75.0,
            context_window=200_000,
            tags=[
                "claude",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
            ],
            notes=(
                "Frontier tier. Slow + expensive, but best for detailed "
                "vision, complex reasoning, audits, security review. "
                "PLACEHOLDER: edit adapter_model_name to 'claude-opus-4-7' "
                "(or the exact Anthropic id) when Opus 4.7 is released."
            ),
        ),
    ]


def find(specs: Iterable[ModelSpec], model_id: str) -> Optional[ModelSpec]:
    for spec in specs:
        if spec.id == model_id:
            return spec
    return None


def enabled_specs(specs: Iterable[ModelSpec]) -> list[ModelSpec]:
    return [s for s in specs if s.enabled]
