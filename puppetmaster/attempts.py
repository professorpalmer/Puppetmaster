"""Additive consumption ledger; independent of selected-result usage accounting.

Records contain only immutable scalars. An attempt identifies one actual invocation,
not a task retry counter or worker. Reuse without execution creates no attempt.
Call from_run once per invocation (supply a distinct invocation_id if a run makes
multiple calls). No runtime path is instrumented by this module.

Observation IDs identify source events within an attempt. Replaying identical
content is a no-op; changing content under the same key is an error. Later or
more complete reports need a new source event ID. Observations are snapshots,
NOT additive deltas: consumers must reconcile overlapping reports before sums.
None means unknown for each metric; explicit measured zero remains zero. Plan
marginal cost and API charges are distinguished by cost_basis. An API-equivalent
estimate for plan usage must use api_equivalent, never api or plan_marginal.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

from puppetmaster.models import AgentRun


class LedgerConflictError(ValueError):
    """An immutable ledger key was reused with different content."""


def _identity(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ledger identities must be nonempty strings")


@dataclass(frozen=True)
class ExecutionAttempt:
    job_id: str
    task_id: str
    run_id: str
    attempt_id: str
    started_at: str
    adapter: str
    model: Optional[str] = None
    provider: Optional[str] = None

    def __post_init__(self) -> None:
        for value in (self.job_id, self.task_id, self.run_id, self.attempt_id,
                      self.started_at, self.adapter):
            _identity(value)
        for value in (self.model, self.provider):
            if value is not None:
                _identity(value)

    @classmethod
    def from_run(cls, run: AgentRun, *, adapter: str,
                 model: Optional[str] = None,
                 invocation_id: Optional[str] = None,
                 provider: Optional[str] = None) -> ExecutionAttempt:
        return cls(run.job_id, run.task_id, run.id,
                   run.id if invocation_id is None else invocation_id,
                   run.started_at, adapter, model, provider)


@dataclass(frozen=True)
class UsageObservation:
    job_id: str
    attempt_id: str
    observation_id: str
    source: str
    observed_at: str
    usage_state: str = "unknown"
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cost_state: str = "unknown"
    cost_usd: Optional[float] = None
    cost_basis: str = "unknown"
    returncode: Optional[int] = None
    timed_out: Optional[bool] = None
    # Source data-quality flags (``partial:<field>``, ``disputed:<field>``,
    # ``conflict:<name>``): the counters are what the source reported, these
    # say which of them a valuation must not trust.
    quality: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.job_id, self.attempt_id, self.observation_id,
                      self.source, self.observed_at):
            _identity(value)
        if not isinstance(self.quality, (list, tuple)):
            raise ValueError("quality must be a sequence of flags")
        for flag in self.quality:
            _identity(flag)
        object.__setattr__(self, "quality", tuple(sorted(set(self.quality))))
        if self.returncode is not None and (type(self.returncode) is not int or not -(2**63) <= self.returncode < 2**63):
            raise ValueError("returncode must be an integer or None")
        if self.timed_out is not None and type(self.timed_out) is not bool:
            raise ValueError("timed_out must be a boolean or None")
        for state in (self.usage_state, self.cost_state):
            if state not in ("unknown", "measured", "estimated"):
                raise ValueError("state must be unknown, measured, or estimated")
        counts = (self.tokens_in, self.tokens_out, self.cache_read_tokens,
                  self.cache_write_tokens)
        for count in counts:
            if count is not None and (type(count) is not int or not 0 <= count < 2**63):
                raise ValueError("token counts must be nonnegative integers or None")
        if (self.usage_state == "unknown") != all(v is None for v in counts):
            raise ValueError("usage state must agree with presence of token counts")
        if self.cost_basis not in ("unknown", "api", "plan_marginal", "api_equivalent"):
            raise ValueError("invalid cost basis")
        if self.cost_usd is not None:
            if (type(self.cost_usd) not in (int, float) or
                    not 0 <= self.cost_usd <= 1_000_000_000_000 or not math.isfinite(self.cost_usd)):
                raise ValueError("cost must be finite and nonnegative or None")
            object.__setattr__(self, "cost_usd", float(self.cost_usd))
        if (self.cost_state == "unknown") != (self.cost_usd is None):
            raise ValueError("cost state must agree with presence of cost")
        if self.cost_usd is not None and self.cost_basis == "unknown":
            raise ValueError("known cost requires a cost basis")
        if self.cost_basis == "api_equivalent" and self.cost_state == "measured":
            raise ValueError("API-equivalent cost is an estimate")


def canonical_record(record: Union[ExecutionAttempt, UsageObservation]) -> str:
    """Stable sorted JSON, explicit nulls, no nonfinite numbers; no redaction loss.

    An empty ``quality`` is omitted so unflagged records stay readable by
    releases that predate the field.
    """
    data = asdict(record)
    if not data.get("quality", True):
        del data["quality"]
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
