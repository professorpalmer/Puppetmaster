"""Invocation accounting, independent of artifact admission and selected usage.

The runtime binds its store/run to this thread. Adapters open an invocation only
after admission, immediately before the CLI or provider call. Observations are
source snapshots, not deltas; shared return paths must not record them again.
"""
from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar

from puppetmaster.attempts import ExecutionAttempt, UsageObservation
from puppetmaster.budget import BudgetAdmissionError, BudgetLiability
from puppetmaster.models import now_iso, new_id

_scope = ContextVar("execution_accounting", default=None)
_dispatch_guard = ContextVar("invocation_dispatch_guard", default=None)
_current = ContextVar("current_invocation", default=None)
# The last invocation that closed in this execution scope: final captures are
# written after the call returns and still belong to that attempt.
_settled = ContextVar("settled_invocation", default=None)
_log = logging.getLogger(__name__)


def check_external_dispatch():
    """Check invocation ownership after preparation, at the external boundary.

    The context-local callback follows synchronous provider/CLI helper calls
    without changing legacy adapter signatures. Unbound calls remain unchanged.
    """
    from puppetmaster.cancellation import check_cancellation
    check_cancellation()
    guard = _dispatch_guard.get()
    if guard is not None:
        guard()


def current_attempt(task):
    """The attempt id ``task``'s captures belong to, or None.

    The open invocation's, else the one that last closed in this execution
    scope, so the captures written after the call returns stay with it.
    """
    task_id = getattr(task, "id", None)
    capture = _current.get()
    if capture is not None:
        return capture.attempt.attempt_id if capture.task.id == task_id else None
    settled = _settled.get()
    if settled is not None and settled[0] == task_id:
        return settled[1]
    return None


@contextmanager
def execution_scope(store, run, task, *, lease_lost=None):
    token = _scope.set((store, run, task, lease_lost))
    settled_token = _settled.set(None)
    from puppetmaster.cancellation import cancellation_scope
    try:
        with cancellation_scope(store, task):
            yield
    finally:
        _settled.reset(settled_token)
        _scope.reset(token)


def _first(data, *keys):
    for key in keys:
        value = data.get(key)
        if type(value) is int and value >= 0:
            return value
    return None


def usage_fields(data):
    """Normalize only present metrics; zero and absent must remain distinct."""
    data = data if isinstance(data, dict) else {}
    details = data.get("input_tokens_details") or data.get("prompt_tokens_details") or {}
    counts = dict(
        tokens_in=_first(data, "tokens_in", "input_tokens", "prompt_tokens", "inputTokens"),
        tokens_out=_first(data, "tokens_out", "output_tokens", "completion_tokens", "outputTokens"),
        cache_read_tokens=_first(data, "cache_read_tokens", "cached_input_tokens", "cache_read_input_tokens", "cached_tokens", "cacheReadTokens", "cacheReadInputTokens", "cacheReadInputTokenCount"),
        cache_write_tokens=_first(data, "cache_write_tokens", "cache_creation_input_tokens", "cacheWriteTokens", "cacheWriteInputTokens", "cacheWriteInputTokenCount"),
    )
    if counts["cache_read_tokens"] is None and isinstance(details, dict):
        counts["cache_read_tokens"] = _first(details, "cached_tokens")
    return dict(counts, usage_state=(
        "unknown" if all(v is None for v in counts.values()) else
        "estimated" if data.get("tokens_estimated") else "measured"
    ))


class Invocation:
    def __init__(self, scope, adapter, model):
        self.store, run, task, self.lease_lost = scope
        self.task = task
        self.billing = (task.payload or {}).get("billing")
        # A fresh launch nonce is linked to its run, never to the resettable
        # retry counter. It is generated once, then reused on persistence retry.
        self.attempt = ExecutionAttempt(
            task.job_id, task.id, run.id, f"{run.id}:{new_id('invoke')}",
            now_iso(), adapter or task.adapter, model or (task.payload or {}).get("model"),
        )
        self.observations = {}
        self.recorded = False
        self.budgeted = False
        self.authoritative = {}
        self.started = time.monotonic()
        self.settlement = None
        self.outcome_complete = True

    def admit(self):
        """Persistence here is mandatory: failure must precede the external call."""
        if self.store.get_job(self.attempt.job_id).budget_policy is None:
            return
        self.budgeted = True
        if self.lease_lost is not None and self.lease_lost():
            raise BudgetAdmissionError("budget dispatch blocked: worker lease lost")
        values = dict((self.task.payload or {}).get("budget_allowance") or {})
        values.setdefault("billing", self.billing or "unknown")
        if values["billing"] != (self.billing or "unknown"):
            raise BudgetAdmissionError("budget allowance billing conflicts with invocation")
        if self.billing == "plan":
            values.setdefault("plan_marginal_usd", 0)
            values.setdefault("cost_state", "known")
        allowance = BudgetLiability(**values)
        with self.store.budget_dispatch_scope(self.attempt.job_id):
            self.store.reserve_dispatch(self.attempt, allowance)
            if self.lease_lost is not None and self.lease_lost():
                self.store.release_undispatched(
                    self.attempt.job_id, self.attempt.attempt_id,
                    non_dispatch_proof="lease lost before adoption and external call")
                raise BudgetAdmissionError("budget dispatch blocked: worker lease lost")
            self.store.adopt_dispatch(self.attempt.job_id, self.attempt.attempt_id,
                                      adoption_id=self.attempt.attempt_id)
            lost_after_adoption = self.lease_lost is not None and self.lease_lost()
            # Persist uncertainty before dispatch, so a killed process cannot leave
            # its original allowance masquerading as complete consumption forever.
            self.store.reconcile_reservation(
                self.attempt.job_id, self.attempt.attempt_id,
                reconciliation_id="dispatch:pending", liability=BudgetLiability(),
                final=False, evidence="dispatch owned; outcome not yet available")
            if lost_after_adoption:
                raise BudgetAdmissionError("budget dispatch blocked: worker lease lost")
            self.check_dispatch_lease()

    def check_dispatch_lease(self):
        if self.budgeted and self.lease_lost is not None and self.lease_lost():
            raise BudgetAdmissionError("budget dispatch blocked: worker lease lost")

    def reconcile(self, completed):
        if not self.budgeted:
            return
        if self.settlement is None:
            completed = completed and self.outcome_complete
            owned = self.lease_lost is None or not self.lease_lost()
            final = completed and owned and len(self.authoritative) == 1
            obs = (next(iter(self.authoritative.values()))
                   if len(self.authoritative) == 1 else None)
            billing = self.billing or "unknown"
            cost = (obs.cost_usd if billing in ("api", "plan") and obs and
                    obs.cost_state == "measured" and
                    obs.cost_basis == ("api" if billing == "api" else "plan_marginal")
                    else None)
            if billing == "plan" and completed and owned:
                cost = 0
            liability = BudgetLiability(
                billing=billing, cost_state=("known" if final else "partial")
                if cost is not None else "unknown",
                api_usd=cost if billing == "api" else None,
                plan_marginal_usd=cost if billing == "plan" else None,
                tokens_in=obs.tokens_in if obs and obs.usage_state == "measured" else None,
                tokens_out=obs.tokens_out if obs and obs.usage_state == "measured" else None,
                elapsed_seconds=time.monotonic() - self.started if completed and owned else None,
            )
            self.settlement = dict(reconciliation_id="dispatch:outcome", liability=liability,
                                   final=final, evidence="invocation authoritative return" if final
                                   else "invocation outcome incomplete or ownership lost")
        try:
            self.store.reconcile_reservation(self.attempt.job_id, self.attempt.attempt_id,
                                             **self.settlement)
        except Exception as exc:
            # The durable pending record remains a fence even if settlement fails.
            self._error("budget.reconciliation_failed", "reconcile", type(exc).__name__)

    def _write(self, operation, record):
        for _ in range(2):
            try:
                getattr(self.store, operation)(record)
                return True
            except Exception as exc:
                error_type = type(exc).__name__
        self._error("consumption.persistence_failed", operation, error_type)
        return False

    def _error(self, event, operation, error_type):
        payload = {"task_id": self.attempt.task_id,
                   "attempt_id": self.attempt.attempt_id,
                   "operation": operation, "error_type": error_type}
        _log.warning("%s: %s", event, payload)
        try:
            self.store.emit(self.attempt.job_id, event, payload)
        except Exception:
            pass

    def observe(self, data=None, *, key="return", source=None, cost_basis="unknown", final=False):
        try:
            self._observe(data, key=key, source=source, cost_basis=cost_basis)
            if final:
                self.authoritative[key] = self.observations[key]
        except Exception as exc:
            # Telemetry cannot replace an adapter exception or erase a result.
            self.authoritative.pop(key, None)
            self.outcome_complete = False
            self._error("consumption.capture_failed", "observe", type(exc).__name__)

    def _observe(self, data, *, key, source, cost_basis):
        data = data if isinstance(data, dict) else {}
        cost = next((data[k] for k in ("real_cost_usd", "cost_usd", "cost", "total_cost_usd")
                     if type(data.get(k)) in (int, float) and
                     math.isfinite(data[k]) and data[k] >= 0), None)
        basis = data.get("cost_basis", cost_basis)
        if basis not in ("api", "plan_marginal", "api_equivalent"):
            basis = "unknown"
        # An amount without billing provenance is not an attributable charge.
        if basis == "unknown":
            cost = None
        observation = UsageObservation(
            self.attempt.job_id, self.attempt.attempt_id, key,
            source or self.attempt.adapter,
            self.observations[key].observed_at if key in self.observations else now_iso(),
            **usage_fields(data),
            cost_usd=cost, cost_basis=basis,
            cost_state=("unknown" if cost is None else
                        "estimated" if basis == "api_equivalent" or data.get("cost_state") == "estimated"
                        else "measured"),
        )
        previous = self.observations.setdefault(key, observation)
        if previous != observation:
            raise ValueError("invocation observation identity has different facts")
        # Retry the same immutable start if its initial write failed.
        if not self.recorded:
            self.recorded = self._write("record_attempt", self.attempt)
        if self.recorded:
            self._write("record_usage_observation", observation)

    def process_exit(self, result):
        observation = UsageObservation(
            self.attempt.job_id, self.attempt.attempt_id, "process:exit",
            self.attempt.adapter, now_iso(),
            returncode=getattr(result, "returncode", None),
            timed_out=getattr(result, "timed_out", None),
        )
        if not self.recorded:
            self.recorded = self._write("record_attempt", self.attempt)
        if self.recorded:
            self._write("record_usage_observation", observation)

    def stdout(self, stdout, attempt_usage=None):
        if isinstance(attempt_usage, dict):
            # The adapter derived this attempt's own usage (a resumed Codex
            # turn.completed is session-cumulative). Unlinked usage stays
            # unknown and never settles as final.
            linked = attempt_usage.get("usage_scope") == "attempt"
            counts = attempt_usage.get("usage") if linked else None
            counts = counts if isinstance(counts, dict) else {}
            self.observe({
                "input_tokens": counts.get("input_tokens"),
                "cached_input_tokens": counts.get("cached_input_tokens"),
                "cache_write_tokens": counts.get("cache_write_input_tokens"),
                "output_tokens": counts.get("output_tokens"),
            }, key="codex:rollout", source=attempt_usage.get("usage_provenance"),
                final=linked, cost_basis=(
                "api_equivalent" if self.billing == "plan" else
                "api" if self.billing == "api" else "unknown"))
        if not isinstance(stdout, str):
            return
        try:
            events = [json.loads(stdout)]
        except ValueError:
            events = []
            for line in stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            if event.get("type") in ("turn.failed", "error"):
                self.outcome_complete = False
            if isinstance(attempt_usage, dict):
                continue
            if not isinstance(event.get("usage"), dict) and "total_cost_usd" not in event:
                continue
            data = dict(event.get("usage") or {})
            if "total_cost_usd" in event:
                data["total_cost_usd"] = event["total_cost_usd"]
            terminal = event.get("type") == "result" or (
                event.get("type") == "turn.completed" and bool(data)
            )
            self.observe(data, key=f"stdout:{index}", final=terminal, cost_basis=(
                "api_equivalent" if self.billing == "plan" else
                "api" if self.billing == "api" else "unknown"
            ))
            if event.get("type") == "turn.completed":
                observation = self.authoritative.get(f"stdout:{index}")
                if observation is not None and observation.cost_state != "measured" and (
                        observation.tokens_in is None or observation.tokens_out is None):
                    self.authoritative.pop(f"stdout:{index}", None)


class _UnboundInvocation:
    def __init__(self, billing):
        self.billing = billing

    def observe(self, *args, **kwargs):
        pass

    def stdout(self, stdout, attempt_usage=None):
        pass


@contextmanager
def invocation(*, adapter=None, model=None, billing=None, source=None):
    scope = _scope.get()
    if scope is None:
        # Direct adapter calls without a runtime have no authoritative store.
        # Do not silently create a second ledger in an inferred project.
        _log.warning("Invocation accounting unbound: adapter=%s", adapter)
        yield _UnboundInvocation(billing)
        return
    capture = Invocation(scope, adapter, model)
    # Launch billing applies to its selected adapter/model, not a fallback
    # invocation on a different target. Legacy unpinned calls retain defaults.
    payload = capture.task.payload or {}
    same_target = (adapter is None or adapter == capture.task.adapter) and (
        model is None or model == payload.get("model"))
    if billing is not None and ("billing" not in payload or not same_target):
        capture.billing = billing
    capture.admit()
    capture.recorded = capture._write("record_attempt", capture.attempt)
    completed = False
    guard_token = _dispatch_guard.set(capture.check_dispatch_lease)
    current_token = _current.set(capture)
    try:
        capture.check_dispatch_lease()
        yield capture
        completed = True
    finally:
        _current.reset(current_token)
        _settled.set((capture.task.id, capture.attempt.attempt_id))
        _dispatch_guard.reset(guard_token)
        if not capture.observations:
            capture.observe(source=f"{source or capture.attempt.adapter}:usage_unavailable")
        if capture.billing == "plan":
            capture.observe({"cost_usd": 0}, key="billing:plan",
                            source="task:billing", cost_basis="plan_marginal")
        capture.reconcile(completed)


def invoke_cli(call, *, accounting_adapter=None, accounting_model=None, **kwargs):
    with invocation(adapter=accounting_adapter, model=accounting_model) as capture:
        result = call(**kwargs)
        if isinstance(capture, Invocation):
            try:
                capture.process_exit(result)
            except Exception as exc:
                capture._error("consumption.capture_failed", "process_exit", type(exc).__name__)
        if isinstance(capture, Invocation) and (
                getattr(result, "timed_out", False) is True or
                getattr(result, "output_limit_hit", False) is True or
                getattr(result, "spawn_error", None) or
                (type(getattr(result, "returncode", None)) is int and result.returncode != 0)):
            capture.outcome_complete = False
        try:
            capture.stdout(result.stdout, getattr(result, "attempt_usage", None))
        except Exception as exc:
            if isinstance(capture, Invocation):
                capture._error("consumption.capture_failed", "stdout", type(exc).__name__)
        return result
