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
* optional ``payload_defaults`` injected into task payloads whenever this
  entry is routed (for adapter-specific knobs such as effort)
* optional ``output_token_multiplier`` for honest cost estimates when a
  variant burns more or fewer output tokens than its base model
"""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from puppetmaster.fs_permissions import write_private_text


REGISTRY_ENV = "PUPPETMASTER_MODELS_PATH"

# Discovery names are user-facing source names, while registry entries use
# adapter names. Keep this mapping in the shared registry layer so diagnostics,
# CLI discovery, and preflight cannot silently disagree about which entries a
# catalog describes.
DISCOVERY_SOURCE_TO_ADAPTER: dict[str, str] = {
    "cursor": "cursor",
    "openai": "openai",
    "anthropic": "claude-code",
    "claude": "claude-code",
    "codex": "codex",
    "hermes": "hermes",
    "agentic": "agentic",
}


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
    payload_defaults: dict[str, Any] = field(default_factory=dict)
    output_token_multiplier: float = 1.0
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
        if self.output_token_multiplier <= 0:
            raise ValueError(
                f"output_token_multiplier for {self.id} must be greater than 0"
            )
        if not isinstance(self.payload_defaults, dict):
            raise ValueError(
                f"payload_defaults for {self.id} must be a JSON object, "
                f"got {type(self.payload_defaults).__name__}"
            )
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
        scaled_tokens_out = tokens_out * self.output_token_multiplier
        return (
            (tokens_in / 1_000_000.0) * self.input_per_mtok_usd
            + (scaled_tokens_out / 1_000_000.0) * self.output_per_mtok_usd
        )

    def marginal_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """Marginal USD this call adds to the user's bill; subscription-covered models add nothing."""
        if self.is_plan_billed:
            return 0.0
        return self.estimate_cost_usd(tokens_in, tokens_out)

    def routing_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """Nominal usage cost used to rank models within a shared plan pool.

        Plan-billed models have zero marginal USD, but Cursor still consumes
        finite first-party usage at model-specific rates. Routing uses the
        nominal rate for quality/cost decisions while accounting continues to
        report the true marginal bill via :meth:`marginal_cost_usd`.
        """
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
    payload: dict[str, Any] = {
        "models": [_spec_to_jsonable(spec) for spec in specs],
    }
    write_private_text(resolved, json.dumps(payload, indent=2, sort_keys=False) + "\n")
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
        ("payload_defaults", {}),
        ("output_token_multiplier", 1.0),
    ]:
        if data.get(k) == default_value:
            data.pop(k)
    return data


def starter_registry() -> list[ModelSpec]:
    """A starter registry organized into the four tiers most users
    actually think in: fast/cheap, balanced, high-quality, frontier.

    Tier IDs (``cursor/composer-2-5``, ``cursor/grok-4-5``,
    ``cursor/gpt-5-6-luna``,
    ``cursor/gpt-5-6-terra``, ``cursor/gpt-5-6-sol``,
    ``claude-code/opus-4-6``, ``claude-code/opus-4-7``,
    ``claude-code/opus-4-8``, ``cursor/claude-fable-5``,
    ``claude-code/fable-5``)
    reflect a common mental model where the cheap tier is Cursor's house
    model, the balanced tier is GPT, the Cursor workhorse is Grok 4.5
    (Opus-class at plan-billed speed/cost), and the tip-of-stack frontier
    is Anthropic Opus / Fable 5. ``adapter_model_name`` values are the
    literal strings each adapter passes through to its SDK / CLI today
    (verified against Cursor's runtime model catalog and Anthropic's
    claude CLI), so the starter registry is callable end-to-end without
    edits.

    Capability scores are normalized routing scores informed by agentic coding
    benchmarks; nominal prices mirror Cursor's public model-rate table. Edit
    either to match a private plan, negotiated pool, or local measurements.
    """
    return [
        ModelSpec(
            id="cursor/composer-2-5",
            adapter="cursor",
            adapter_model_name="composer-2.5",
            capability_score=55,
            input_per_mtok_usd=0.5,
            output_per_mtok_usd=2.5,
            context_window=0,
            billing="plan",
            tags=["tools", "cursor", "cheapest", "cheap", "fast", "reading", "code", "workhorse"],
            notes=(
                "Cheapest Cursor tier and default workhorse for grunt work: "
                "verification, exploration, formatting, cleanup, and other "
                "low-stakes tasks. It is intentionally not the quality choice. "
                "Cursor plan usage is subscription-covered; the 'cheapest' tag "
                "captures relative plan/credit preference, not marginal USD."
            ),
        ),
        ModelSpec(
            id="cursor/gpt-5-6-luna",
            adapter="cursor",
            adapter_model_name="gpt-5.6-luna",
            capability_score=85,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=6.0,
            context_window=1_050_000,
            billing="plan",
            tags=["tools", "cursor", "affordable", "balanced", "fast", "vision", "code"],
            notes=(
                "Affordable GPT-5.6 Cursor tier. Prefer for capable everyday "
                "implementation and subagent work before Terra/Sol; plan-billed "
                "with no marginal API USD."
            ),
        ),
        ModelSpec(
            id="cursor/gpt-5-6-terra",
            adapter="cursor",
            adapter_model_name="gpt-5.6-terra",
            capability_score=94,
            input_per_mtok_usd=2.5,
            output_per_mtok_usd=15.0,
            context_window=1_050_000,
            billing="plan",
            tags=["tools", "cursor", "expensive", "quality", "vision", "code", "reasoning"],
            notes=(
                "Pretty-expensive GPT-5.6 Cursor quality tier. Reserve for "
                "difficult reviews and implementation where Luna/Grok are not "
                "enough; plan-billed with no marginal API USD."
            ),
        ),
        ModelSpec(
            id="cursor/gpt-5-6-sol",
            adapter="cursor",
            adapter_model_name="gpt-5.6-sol",
            capability_score=99,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_050_000,
            billing="plan",
            tags=["tools", "cursor", "very-expensive", "frontier", "vision", "code", "reasoning"],
            notes=(
                "Very-expensive GPT-5.6 Sol Cursor frontier tier. Reserve for "
                "the hardest reasoning and codebase-scale work; plan-billed with "
                "no marginal API USD."
            ),
        ),
        ModelSpec(
            id="cursor/grok-4-5",
            adapter="cursor",
            adapter_model_name="grok-4.5",
            capability_score=97,
            input_per_mtok_usd=2.0,
            output_per_mtok_usd=6.0,
            context_window=0,
            billing="plan",
            # Live SDK identity is grok-4.5; High+Fast is the catalog default
            # variant (params), not an expanded alias like cursor-grok-4.5-high-fast.
            payload_defaults={
                "params": [
                    {"id": "effort", "value": "high"},
                    {"id": "fast", "value": "true"},
                ]
            },
            tags=[
                "tools",
                "cursor",
                "xai",
                "frontier",
                "fast",
                "code",
                "reasoning",
                "agentic",
                "workhorse",
                "effort:high",
                "param:fast",
            ],
            notes=(
                "Cursor workhorse (released 2026-07-08). SpaceXAI Grok 4.5 via "
                "the Cursor SDK — trained alongside Cursor; ~80 TPS with ~2x "
                "token efficiency vs Opus-class peers. CursorBench 3.2 High "
                "scores 66.7% (above Opus 4.8 Max at 62.3%, below Fable 5 Max "
                "at 70.5%) at ~$1.51/task vs Opus Max ~$5.77 and Fable Max "
                "~$17.32; Cursor notes a training-data asterisk on that "
                "leaderboard. API list price is $2/$6 per MTok, but this "
                "entry is plan-billed ($0 marginal). capability_score=97 sits "
                "just under Opus 4.7/4.8 so balanced routing prefers Grok for "
                "most hard Cursor work while reserving Opus/Fable for the "
                "tip-of-stack. Absorbs the prior Opus-class workhorse band on "
                "Cursor when Fable is available for the absolute hardest tasks. "
                "Default variant is High+Fast via payload_defaults.params."
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
            tags=["tools", "claude", "cheap", "fast", "vision", "reading", "code"],
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
            tags=["tools", "claude", "quality", "vision", "code", "reasoning"],
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
            tags=["tools", 
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
            tags=["tools", 
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
            id="cursor/claude-fable-5",
            adapter="cursor",
            adapter_model_name="claude-fable-5",
            capability_score=100,
            input_per_mtok_usd=10.0,
            output_per_mtok_usd=50.0,
            context_window=0,
            billing="plan",
            tags=["tools", 
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
                "plan, but it is the most expensive Cursor usage tier. "
                "SOTA on CursorBench; capability_score=100 makes it the "
                "default pick for the hardest tasks when your plan exposes "
                "claude-fable-5."
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
            tags=["tools", 
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
        # the publicly-listed GPT-5.4 / GPT-5.5 / GPT-5.6 catalog.
        ModelSpec(
            id="openai/gpt-5-6-sol",
            adapter="openai",
            adapter_model_name="gpt-5.6-sol",
            capability_score=99,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_050_000,
            billing="api",
            tags=["tools", 
                "openai",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
            notes=(
                "OpenAI GPT-5.6 Sol — frontier coding/reasoning flagship "
                "(API id gpt-5.6-sol, alias gpt-5.6). 1.05M context, 128K "
                "max output. Some accounts still return limited-preview 404 "
                "until rollout completes; the entry lights up when access "
                "lands. Routes through the openai adapter (OPENAI_API_KEY)."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-6",
            adapter="openai",
            adapter_model_name="gpt-5.6",
            capability_score=99,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_050_000,
            billing="api",
            tags=["tools", 
                "openai",
                "frontier",
                "vision",
                "detailed-vision",
                "reasoning",
                "code",
                "long-context",
            ],
            notes=(
                "Alias for GPT-5.6 Sol (adapter_model_name gpt-5.6 routes to "
                "the same flagship as openai/gpt-5-6-sol). Same pricing and "
                "capability; use when your client or docs reference gpt-5.6."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-6-terra",
            adapter="openai",
            adapter_model_name="gpt-5.6-terra",
            capability_score=97,
            input_per_mtok_usd=2.5,
            output_per_mtok_usd=15.0,
            context_window=1_050_000,
            billing="api",
            tags=["tools", 
                "openai",
                "quality",
                "vision",
                "code",
                "reasoning",
                "long-context",
            ],
            notes=(
                "OpenAI GPT-5.6 Terra — balanced tier. 1.05M context. "
                "Competitive with GPT-5.5 at half the output cost. Some "
                "accounts still return limited-preview 404 until rollout "
                "completes."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-6-luna",
            adapter="openai",
            adapter_model_name="gpt-5.6-luna",
            capability_score=90,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=6.0,
            context_window=1_050_000,
            billing="api",
            tags=["tools", "openai", "balanced", "fast", "vision", "code", "long-context"],
            notes=(
                "OpenAI GPT-5.6 Luna — cheap/fast tier. 1.05M context. "
                "Strong value for implementation and subagent work. Some "
                "accounts still return limited-preview 404 until rollout "
                "completes."
            ),
        ),
        ModelSpec(
            id="openai/gpt-5-5",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            capability_score=96,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_000_000,
            billing="api",
            tags=["tools", 
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
            tags=["tools", 
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
            tags=["tools", "openai", "balanced", "fast", "vision", "code"],
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
            tags=["tools", "openai", "cheap", "fast", "reading"],
            notes=(
                "OpenAI nano tier. Cheapest member of the GPT-5 family for "
                "high-throughput reading, classification, and trivial edits. "
                "Pricing is an estimate; update once nano pricing is public."
            ),
        ),
        ModelSpec(
            id="codex/gpt-5-6-sol",
            adapter="codex",
            adapter_model_name="gpt-5.6-sol",
            capability_score=100,
            input_per_mtok_usd=5.0,
            output_per_mtok_usd=30.0,
            context_window=1_050_000,
            tags=["tools", 
                "codex",
                "frontier",
                "vision",
                "reasoning",
                "code",
                "agent-loop",
                "long-context",
            ],
            notes=(
                "OpenAI Codex CLI driving gpt-5.6-sol. Same underlying model "
                "and billing as openai/gpt-5-6-sol, but ships an in-CLI agent "
                "loop. Capability_score is 1 higher than openai/gpt-5-6-sol to "
                "reflect the agent-loop advantage on multi-file refactors."
            ),
        ),
        ModelSpec(
            id="codex/gpt-5-6-terra",
            adapter="codex",
            adapter_model_name="gpt-5.6-terra",
            capability_score=98,
            input_per_mtok_usd=2.5,
            output_per_mtok_usd=15.0,
            context_window=1_050_000,
            tags=["tools", 
                "codex",
                "quality",
                "vision",
                "code",
                "reasoning",
                "agent-loop",
                "long-context",
            ],
            notes=(
                "Codex CLI driving gpt-5.6-terra. Balanced tier with the Codex "
                "agent loop enabled. Capability_score is 1 higher than "
                "openai/gpt-5-6-terra for the same agent-loop reason as "
                "codex/gpt-5-6-sol."
            ),
        ),
        ModelSpec(
            id="codex/gpt-5-6-luna",
            adapter="codex",
            adapter_model_name="gpt-5.6-luna",
            capability_score=91,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=6.0,
            context_window=1_050_000,
            tags=["tools", "codex", "balanced", "vision", "code", "agent-loop", "long-context"],
            notes=(
                "Codex CLI driving gpt-5.6-luna. Cheap/fast tier with the "
                "Codex agent loop. Capability_score is 1 higher than "
                "openai/gpt-5-6-luna."
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
            tags=["tools", 
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
            tags=["tools", "codex", "balanced", "vision", "code", "agent-loop"],
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
    model_ids: Optional[Iterable[str]] = None,
    catalog_hash: Optional[str] = None,
    mode: str = "apply",
    pending_diff: Optional[dict[str, Any]] = None,
) -> Path:
    """Record that ``source`` (e.g. ``cursor``/``openai``/``anthropic``) was
    just discovered, with how many models it returned and when.

    ``model_ids`` is an optional membership snapshot. Keeping it beside the
    timestamp lets diagnostics distinguish an old catalog from a registry
    whose entries have drifted away from the last known live catalog.
    """
    from datetime import datetime, timezone

    if mode not in {"apply", "probe"}:
        raise ValueError("discovery metadata mode must be 'apply' or 'probe'")
    path = discovery_meta_path(registry_path)
    meta = read_discovery_meta(registry_path)
    stamp = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry: dict[str, Any] = {"refreshed_at": stamp, "count": count}
    if model_ids is not None:
        normalized_ids = sorted(
            {str(model_id) for model_id in model_ids if str(model_id)}
        )
        entry["model_ids"] = normalized_ids
        entry["catalog_hash"] = catalog_hash or catalog_membership_hash(normalized_ids)
        previous = meta.get(source)
        previous_applied = (
            previous.get("last_applied_hash")
            if isinstance(previous, dict)
            else None
        )
        if mode == "apply":
            entry["last_applied_hash"] = entry["catalog_hash"]
            entry["pending_diff"] = {}
        else:
            entry["last_applied_hash"] = (
                previous_applied
                or (previous.get("catalog_hash") if isinstance(previous, dict) else None)
            )
            entry["pending_diff"] = pending_diff or {}
    if mode == "probe":
        entry["probe_status"] = "ok"
        entry["mode"] = "probe"
    else:
        entry["mode"] = "apply"
    meta[source] = entry
    write_private_text(path, json.dumps(meta, indent=2) + "\n")
    return path


def catalog_membership_hash(model_ids: Iterable[str]) -> str:
    """Return a stable hash for a source's normalized model membership."""
    normalized = sorted({str(model_id) for model_id in model_ids if str(model_id)})
    return "sha256:" + hashlib.sha256(
        "\n".join(normalized).encode("utf-8")
    ).hexdigest()


def catalog_content_hash(catalog: Iterable[object]) -> str:
    """Return a stable hash for full catalog records, including pricing metadata."""
    normalized = json.dumps(
        list(catalog),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def discovery_registry_diff(
    specs: Iterable[ModelSpec],
    source: str,
    model_ids: Iterable[str],
) -> dict[str, Any]:
    """Describe catalog membership not represented by enabled registry entries."""
    adapter = DISCOVERY_SOURCE_TO_ADAPTER.get(source, source)
    registered = {
        spec.adapter_model_name
        for spec in specs
        if spec.adapter == adapter and spec.enabled
    }
    discovered = {str(model_id) for model_id in model_ids if str(model_id)}
    added = sorted(discovered - registered)
    removed = sorted(registered - discovered)
    return {
        "source": source,
        "adapter": adapter,
        "status": "drift" if added or removed else "match",
        "added": added,
        "removed": removed,
    }


def discovery_catalog_changed(
    meta: dict[str, Any], source: str
) -> bool:
    """Whether a probe observed a catalog different from the last apply."""
    entry = meta.get(source)
    if not isinstance(entry, dict):
        return False
    current = entry.get("catalog_hash")
    applied = entry.get("last_applied_hash")
    if current and applied:
        return current != applied
    return bool(entry.get("pending_diff"))


def discovery_registry_drift(
    registry_path: Optional[Path] = None,
    *,
    source: str = "cursor",
) -> dict[str, Any]:
    """Compare enabled registry entries with the last discovered catalog.

    Returns ``status='unknown'`` when the discovery sidecar predates
    membership snapshots, ``status='match'`` when the two sets agree, or
    ``status='drift'`` with stale/new model names when they do not.
    """
    meta = read_discovery_meta(registry_path)
    entry = meta.get(source)
    if not isinstance(entry, dict) or not isinstance(entry.get("model_ids"), list):
        return {"status": "unknown", "source": source}

    adapter = DISCOVERY_SOURCE_TO_ADAPTER.get(source, source)
    try:
        registered = {
            spec.adapter_model_name
            for spec in load_registry(registry_path)
            if spec.adapter == adapter and spec.enabled
        }
    except (RuntimeError, OSError):
        return {"status": "unknown", "source": source}
    discovered = {str(model_id) for model_id in entry["model_ids"]}
    stale = sorted(registered - discovered)
    unregistered = sorted(discovered - registered)
    return {
        "status": "drift" if stale or unregistered else "match",
        "source": source,
        "stale_registry_models": stale,
        "unregistered_catalog_models": unregistered,
    }


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


def _normalize_model_token(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


@dataclass(frozen=True)
class ResolvedModelPin:
    """A user model pin resolved to one enabled registry entry."""

    registry_id: str
    adapter_model_name: str
    adapter: str
    spec: ModelSpec


class AmbiguousModelPinError(ValueError):
    """Raised when a model pin matches more than one enabled registry entry."""


def resolve_model_pin(
    model: str,
    registry: Iterable[ModelSpec],
    *,
    adapter: Optional[str] = None,
) -> Optional[ResolvedModelPin]:
    """Resolve a durable registry id or adapter model name to one registry entry.

    Accepts both ``cursor/grok-4-5`` and ``grok-4.5`` (and slug-equivalent
    forms such as ``grok-4-5``) and returns the canonical registry id plus the
    adapter/provider model name to pass through to the underlying SDK/CLI.
    """
    needle = (model or "").strip()
    if not needle:
        return None

    token = _normalize_model_token(needle.split("/")[-1])
    candidates: list[ModelSpec] = []
    for spec in enabled_specs(registry):
        if adapter is not None and spec.adapter != adapter:
            continue
        if spec.id == needle or spec.adapter_model_name == needle:
            candidates.append(spec)
            continue
        if token and (
            _normalize_model_token(spec.adapter_model_name) == token
            or _normalize_model_token(spec.id.split("/")[-1]) == token
        ):
            candidates.append(spec)

    if not candidates:
        return None
    if len(candidates) > 1:
        ids = ", ".join(sorted(spec.id for spec in candidates))
        raise AmbiguousModelPinError(
            f"model pin {model!r} is ambiguous across enabled registry entries: {ids}"
        )
    spec = candidates[0]
    return ResolvedModelPin(
        registry_id=spec.id,
        adapter_model_name=spec.adapter_model_name,
        adapter=spec.adapter,
        spec=spec,
    )


def stamp_resolved_model_pin(payload: dict, pin: ResolvedModelPin) -> dict:
    """Persist both the canonical registry id and adapter model name."""
    return {
        **payload,
        "model": pin.adapter_model_name,
        "router_model_id": pin.registry_id,
        "pinned_model": pin.registry_id,
        "pinned_adapter_model_name": pin.adapter_model_name,
    }


def apply_cursor_model_pin(
    payload: dict,
    model: str,
    *,
    registry: Optional[Iterable[ModelSpec]] = None,
) -> dict:
    """Stamp an explicit Cursor pin into a durable task payload.

    Adapter dispatch receives ``grok-4.5`` while ``pinned_model`` /
    ``router_model_id`` keep the canonical registry id for cost and audit.
    Ambiguous pins raise :class:`AmbiguousModelPinError` (fail closed) rather
    than being forwarded as a raw model string.
    """
    pin = resolve_model_pin(
        model,
        registry if registry is not None else load_registry(),
        adapter="cursor",
    )
    if pin is None:
        return {**payload, "model": model}
    return stamp_resolved_model_pin(payload, pin)
