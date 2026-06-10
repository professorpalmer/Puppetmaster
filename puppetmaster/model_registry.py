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
    # Billing source for routing cost-containment. ``plan`` = covered by a
    # subscription the user already pays for (Cursor plan, Claude Max,
    # ChatGPT/Codex sub) so marginal spend stays inside that package;
    # ``api`` = billed per-token to a raw provider key (out-of-pocket);
    # ``unknown`` = depends on runtime auth and should be resolved by
    # :mod:`puppetmaster.platform_billing` detection. The router prefers
    # ``plan`` at sufficient capability unless the caller opts into API spend.
    billing: str = "unknown"

    _VALID_BILLING = ("plan", "api", "unknown")

    def __post_init__(self) -> None:
        if not 0 <= self.capability_score <= 100:
            raise ValueError(
                f"capability_score for {self.id} must be 0..100, got {self.capability_score}"
            )
        if self.input_per_mtok_usd < 0 or self.output_per_mtok_usd < 0:
            raise ValueError(f"per-token cost for {self.id} must be non-negative")
        if self.billing not in self._VALID_BILLING:
            raise ValueError(
                f"billing for {self.id} must be one of {self._VALID_BILLING}, got {self.billing!r}"
            )

    @property
    def is_plan_billed(self) -> bool:
        """True when this model is covered by a subscription (no marginal API spend)."""
        return self.billing == "plan"

    def estimate_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """USD cost estimate for one call. Linear; ignores caching / batching."""
        return (
            (tokens_in / 1_000_000.0) * self.input_per_mtok_usd
            + (tokens_out / 1_000_000.0) * self.output_per_mtok_usd
        )

    def marginal_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """Marginal USD this call adds to the user's bill; subscription-covered models add nothing."""
        if self.is_plan_billed:
            return 0.0
        return self.estimate_cost_usd(tokens_in, tokens_out)


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
        ("billing", "unknown"),
    ]:
        if data.get(k) == default_value:
            data.pop(k)
    return data


def starter_registry() -> list[ModelSpec]:
    """A starter registry organized into the four tiers most users
    actually think in: fast/cheap, balanced, high-quality, frontier.

    Tier IDs (``cursor/composer-2-5``, ``cursor/gpt-5-5``,
    ``claude-code/opus-4-6``, ``claude-code/opus-4-7``,
    ``claude-code/opus-4-8``, ``cursor/fable-5``, ``claude-code/fable-5``)
    reflect a common mental model where the cheap tier is Cursor's house
    model, the balanced tier is GPT, and the frontier tiers are Anthropic
    Opus — with Claude Fable 5 as the current top-of-stack flagship above
    Opus 4.8. ``adapter_model_name`` values are the literal strings each
    adapter passes through to its SDK / CLI today (verified against
    Cursor's runtime model catalog and Anthropic's claude CLI), so
    the starter registry is callable end-to-end without edits.

    Capability scores and prices are user-asserted starting points —
    edit them to match your subscriptions.
    """
    return [
        ModelSpec(
            id="cursor/composer-2-5",
            adapter="cursor",
            adapter_model_name="composer-2.5",
            capability_score=55,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
            context_window=0,
            billing="plan",
            tags=["cursor", "cheap", "fast", "reading", "code"],
            notes=(
                "Fast/cheap tier. Cursor's house Composer model — use "
                "for verification, exploration, formatting, and other "
                "low-stakes work. Pricing is $0 because Composer is "
                "bundled with your Cursor plan."
            ),
        ),
        ModelSpec(
            id="cursor/gpt-5-5",
            adapter="cursor",
            adapter_model_name="gpt-5.5",
            capability_score=78,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
            context_window=0,
            billing="plan",
            tags=["cursor", "balanced", "fast", "vision"],
            notes=(
                "Balanced tier. GPT-5.5 via the Cursor SDK — same model "
                "as the openai/gpt-5-5 entry but billed through your "
                "Cursor plan (so $0 from the router's perspective). The "
                "router prefers this over openai/gpt-5-5 under the "
                "balanced policy whenever both qualify."
            ),
        ),
        ModelSpec(
            id="claude-code/haiku-4-5",
            adapter="claude-code",
            adapter_model_name="claude-haiku-4-5",
            capability_score=55,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=5.0,
            context_window=200_000,
            tags=["claude", "cheap", "fast", "vision", "reading", "code"],
            notes=(
                "Cheap/fast tier. Anthropic Haiku 4.5 via the Claude Code "
                "CLI — the Anthropic-side counterpart to Cursor's "
                "composer-2.5, except it bills per token ($1/$5 per MTok) "
                "instead of rolling into a subscription. The router prefers "
                "cursor/composer-2-5 ($0) or openai/gpt-5-4-nano ($0.15/"
                "$0.90) over this entry whenever they exist in the "
                "registry; this entry ensures Claude-Code-only users still "
                "get a cheap-tier routing option instead of falling "
                "through to Opus on trivial tasks."
            ),
        ),
        ModelSpec(
            id="claude-code/opus-4-6",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-6",
            capability_score=88,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=25.0,
            context_window=200_000,
            tags=["claude", "quality", "vision", "code", "reasoning"],
            notes=(
                "High-quality tier. Anthropic Opus 4.6 via the Claude "
                "Code CLI. Workhorse for implementation, refactoring, "
                "and review when you want Anthropic-grade reasoning "
                "without the frontier price. Pricing reflects the "
                "Anthropic 4.x rate schedule ($5/$25 per MTok), not the "
                "older Opus 4.1 rate ($15/$75) the starter registry "
                "shipped before v0.6.3."
            ),
        ),
        ModelSpec(
            id="claude-code/opus-4-7",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-7",
            capability_score=98,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=25.0,
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
                "Previous frontier flagship. Anthropic Opus 4.7 via the "
                "Claude Code CLI. Superseded by claude-code/opus-4-8 (same "
                "price, better benchmarks, 4x larger context) — kept in the "
                "registry so existing routing configs and cost history stay "
                "valid. Pricing reflects the Anthropic 4.x rate schedule "
                "($5/$25 per MTok)."
            ),
        ),
        ModelSpec(
            id="claude-code/opus-4-8",
            adapter="claude-code",
            adapter_model_name="claude-opus-4-8",
            capability_score=99,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=25.0,
            context_window=1_000_000,
            tags=[
                "claude",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
            notes=(
                "Previous frontier flagship. Anthropic Opus 4.8 via the "
                "Claude Code CLI, released 2026-05-28. Builds on Opus 4.7 with "
                "across-the-board benchmark gains, a 1M-token context window, "
                "and codebase-scale parallel-subagent work — at the SAME $5/$25 "
                "per-MTok price as 4.7. Superseded by claude-code/fable-5 "
                "(capability_score=100) for the absolute-hardest tasks; kept so "
                "existing routing configs and cost history stay valid and it "
                "remains the best fallback when Fable 5 is unavailable on an "
                "account. A faster, pricier 'fast mode' ($10/$50 per MTok) "
                "exists for latency-sensitive work — add it as a separate "
                "entry if you want the router to consider it."
            ),
        ),
        ModelSpec(
            id="cursor/fable-5",
            adapter="cursor",
            adapter_model_name="fable-5",
            capability_score=100,
            input_per_mtok_usd=0.0,
            output_per_mtok_usd=0.0,
            context_window=0,
            billing="plan",
            tags=[
                "cursor",
                "frontier",
                "mythos-class",
                "reasoning",
                "code",
                "long-context",
                "vision",
                "detailed-vision",
            ],
            notes=(
                "Frontier flagship on Cursor. Anthropic Claude Fable 5 via the "
                "Cursor SDK (released 2026-06-09). Billed through your Cursor "
                "plan ($0 marginal from the router's perspective). SOTA on "
                "CursorBench; capability_score=100 makes it the default pick "
                "for the hardest tasks when your plan exposes fable-5."
            ),
        ),
        ModelSpec(
            id="claude-code/fable-5",
            adapter="claude-code",
            adapter_model_name="claude-fable-5",
            capability_score=100,
            input_per_mtok_usd=10.0,
            output_per_mtok_usd=50.0,
            context_window=1_000_000,
            billing="unknown",
            tags=[
                "claude",
                "frontier",
                "mythos-class",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
            notes=(
                "Frontier flagship. Anthropic Claude Fable 5 via the Claude "
                "Code CLI (API id claude-fable-5, released 2026-06-09). "
                "Subscription access is staged and ends 2026-06-22; many "
                "enterprise plans do not include it yet — expect "
                "model_unavailable and auto-fallback to claude-code/opus-4-8 "
                "or a plan-billed cursor/* alternate when absent. Pricing "
                "reflects the public fast-mode rate ($10/$50 per MTok)."
            ),
        ),
        # OpenAI tier — uses the openai adapter directly with OPENAI_API_KEY,
        # bypassing Cursor's SDK entirely. Pricing and model IDs are
        # the publicly-listed GPT-5.4 / GPT-5.5 catalog.
        ModelSpec(
            id="openai/gpt-5-5",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            capability_score=96,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_000_000,
            billing="api",
            tags=[
                "openai",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
            notes=(
                "OpenAI frontier coding/reasoning model. 1M context, 128K "
                "max output. Comparable capability to Opus 4.7 but cheaper. "
                "Routes through the openai adapter (needs OPENAI_API_KEY)."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-4",
            adapter="openai",
            adapter_model_name="gpt-5.4",
            capability_score=86,
            input_per_mtok_usd=2.5,
            output_per_mtok_usd=15.0,
            context_window=1_000_000,
            billing="api",
            tags=[
                "openai",
                "quality",
                "fast",
                "vision",
                "code",
                "reasoning",
                "long-context",
            ],
            notes=(
                "OpenAI workhorse model. Same 1M context as GPT-5.5 at half "
                "the price. Good default for implementation tasks where you "
                "don't need frontier reasoning."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-4-mini",
            adapter="openai",
            adapter_model_name="gpt-5.4-mini",
            capability_score=70,
            input_per_mtok_usd=0.75,
            output_per_mtok_usd=4.5,
            context_window=400_000,
            billing="api",
            tags=["openai", "balanced", "fast", "vision", "code"],
            notes=(
                "OpenAI mini for coding, computer use, and subagents. 400K "
                "context. Cheap enough to run as a default for exploration "
                "and verification while staying capable."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-4-nano",
            adapter="openai",
            adapter_model_name="gpt-5.4-nano",
            capability_score=52,
            input_per_mtok_usd=0.15,
            output_per_mtok_usd=0.9,
            context_window=400_000,
            billing="api",
            tags=["openai", "cheap", "fast", "reading"],
            notes=(
                "OpenAI nano tier. Cheapest member of the GPT-5 family for "
                "high-throughput reading, classification, and trivial edits. "
                "Pricing is an estimate; update once nano pricing is public."
            ),
        ),
        ModelSpec(
            id="codex/gpt-5-5",
            adapter="codex",
            adapter_model_name="gpt-5.5",
            capability_score=97,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_000_000,
            tags=[
                "codex",
                "frontier",
                "vision",
                "reasoning",
                "code",
                "agent-loop",
                "long-context",
            ],
            notes=(
                "OpenAI Codex CLI (`codex exec --json`) driving gpt-5.5. "
                "Same underlying model and billing as openai/gpt-5-5, but "
                "ships an in-CLI agent loop (file edits, shell, search) so "
                "the model can act, not just answer. Capability_score is "
                "set 1 higher than openai/gpt-5-5 to reflect the agent-loop "
                "advantage on multi-file refactors and codebase audits; for "
                "pure one-shot reasoning, prefer openai/gpt-5-5 (cheaper "
                "per-task because no tool-use round-trips)."
            ),
        ),
        ModelSpec(
            id="codex/gpt-5-4-mini",
            adapter="codex",
            adapter_model_name="gpt-5.4-mini",
            capability_score=72,
            input_per_mtok_usd=0.75,
            output_per_mtok_usd=4.5,
            context_window=400_000,
            tags=["codex", "balanced", "vision", "code", "agent-loop"],
            notes=(
                "Codex CLI driving gpt-5.4-mini. Mini-tier coding agent "
                "with the same per-token cost as openai/gpt-5-4-mini, but "
                "with the Codex agent loop enabled (file edits, shell, "
                "search). Slightly higher capability_score than its "
                "openai/* counterpart for the same reason as codex/gpt-5-5."
            ),
        ),
    ]


def discovery_meta_path(registry_path: Optional[Path] = None) -> Path:
    """Sidecar file that records when each catalog source was last discovered.

    Kept separate from ``models.json`` so discovery bookkeeping never perturbs
    the hand-editable registry (or its round-trip tests)."""
    resolved = registry_path or default_registry_path()
    return resolved.with_name(resolved.stem + ".discovery.json")


def read_discovery_meta(registry_path: Optional[Path] = None) -> dict[str, Any]:
    """Return ``{source: {refreshed_at, count}}`` recorded by ``models discover``."""
    path = discovery_meta_path(registry_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_discovery_meta(
    source: str,
    count: int,
    registry_path: Optional[Path] = None,
    *,
    now_iso: Optional[str] = None,
) -> Path:
    """Record that ``source`` (e.g. ``cursor``/``openai``/``anthropic``) was
    just discovered, with how many models it returned and when."""
    from datetime import datetime, timezone

    path = discovery_meta_path(registry_path)
    meta = read_discovery_meta(registry_path)
    stamp = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta[source] = {"refreshed_at": stamp, "count": count}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def catalog_staleness_days(
    meta: dict[str, Any], source: str, *, now: Optional["object"] = None
) -> Optional[float]:
    """Age in days since ``source`` was last discovered, or None if never."""
    from datetime import datetime, timezone

    entry = meta.get(source) or {}
    refreshed = entry.get("refreshed_at")
    if not refreshed:
        return None
    try:
        when = datetime.strptime(refreshed, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - when).total_seconds() / 86400.0)


def enabled_specs(specs: Iterable[ModelSpec]) -> list[ModelSpec]:
    return [s for s in specs if s.enabled]
