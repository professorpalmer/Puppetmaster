"""Optional OpenTelemetry tracing for Puppetmaster swarms.

Zero-cost and zero-dependency by default: nothing here imports
``opentelemetry`` or does any work unless the operator opts in via the
standard env vars. When enabled, each finished job emits a trace —

    puppetmaster.job            (session span: goal, status, totals)
    └── puppetmaster.task       (one child per worker: role, adapter, model,
                                 status, estimated cost, gen_ai.* attributes)

— built from the durable job/task/artifact records and exported via OTLP (or
printed to the console). Because Puppetmaster runs workers as independent OS
subprocesses, the trace is assembled from records at job completion rather
than streamed live; the result is a faithful, reliable per-worker tree in
Jaeger/Datadog/etc. without threading span context across process boundaries.

Enable it the same way GitAgent / OTel SDKs do:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # turns tracing on
    OTEL_TRACES_EXPORTER=console                          # or print to stdout
    PUPPETMASTER_OTEL_ENABLED=false                       # force-off override
    OTEL_SERVICE_NAME=puppetmaster                        # resource name

Install the optional dependency with ``pip install 'puppetmaster-ai[otel]'``.
If tracing is enabled but the SDK isn't installed, Puppetmaster warns once and
continues — telemetry never breaks a run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from puppetmaster.models import Artifact, ArtifactType, Job, Task, parse_iso

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_warned_missing_sdk = False


def telemetry_enabled(env: Optional[dict] = None) -> bool:
    """True when the operator has opted into tracing.

    On unless an explicit override says otherwise: ``PUPPETMASTER_OTEL_ENABLED``
    wins if set; otherwise an OTLP endpoint or the console exporter turns it on.
    """
    env = env if env is not None else os.environ
    forced = env.get("PUPPETMASTER_OTEL_ENABLED")
    if forced is not None:
        if forced.strip().lower() in _FALSE:
            return False
        if forced.strip().lower() in _TRUE:
            return True
    if env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return True
    if (env.get("OTEL_TRACES_EXPORTER") or "").strip().lower() == "console":
        return True
    return False


@dataclass(frozen=True)
class TaskSpan:
    """A flattened, OTel-agnostic description of one worker's span."""

    task_id: str
    role: str
    adapter: str
    model: Optional[str]
    status: str
    start_iso: Optional[str]
    end_iso: Optional[str]
    cost_usd: float = 0.0
    failure: Optional[str] = None

    def attributes(self) -> dict:
        attrs = {
            "puppetmaster.task.id": self.task_id,
            "puppetmaster.task.role": self.role,
            "puppetmaster.task.status": self.status,
            "gen_ai.system": self.adapter,
            "gitagent.cost_usd": self.cost_usd,
            "puppetmaster.cost_usd": self.cost_usd,
        }
        if self.model:
            attrs["gen_ai.request.model"] = self.model
        if self.failure:
            attrs["puppetmaster.task.failure"] = self.failure
        return attrs


@dataclass(frozen=True)
class JobTrace:
    """A flattened, OTel-agnostic description of a whole job trace."""

    job_id: str
    goal: str
    status: str
    start_iso: Optional[str]
    end_iso: Optional[str]
    tasks: list[TaskSpan] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(t.cost_usd for t in self.tasks), 6)

    def attributes(self) -> dict:
        statuses: dict[str, int] = {}
        for task in self.tasks:
            statuses[task.status] = statuses.get(task.status, 0) + 1
        attrs = {
            "puppetmaster.job.id": self.job_id,
            "puppetmaster.job.goal": self.goal[:300],
            "puppetmaster.job.status": self.status,
            "puppetmaster.job.task_count": len(self.tasks),
            "puppetmaster.cost_usd": self.total_cost_usd,
            "gitagent.cost_usd": self.total_cost_usd,
        }
        for status, count in statuses.items():
            attrs[f"puppetmaster.job.tasks.{status}"] = count
        return attrs


def build_job_trace(
    job: Job,
    tasks: list[Task],
    artifacts: list[Artifact],
) -> JobTrace:
    """Assemble an OTel-agnostic :class:`JobTrace` from durable records.

    Pure and dependency-free so it is fully unit-testable without the OTel SDK.
    Per-task model + cost are read from the ROUTING artifact when present
    (falling back to the task payload), and any worker failure class is pulled
    from a blocked/failed verification artifact.
    """
    routing_by_task: dict[str, Artifact] = {}
    failure_by_task: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.type == ArtifactType.ROUTING:
            routing_by_task[artifact.task_id] = artifact
        else:
            failure = (artifact.payload or {}).get("failure")
            if failure and artifact.task_id not in failure_by_task:
                failure_by_task[artifact.task_id] = str(failure)

    task_spans: list[TaskSpan] = []
    for task in tasks:
        routing = routing_by_task.get(task.id)
        payload = task.payload or {}
        model = None
        cost = 0.0
        if routing is not None:
            model = routing.payload.get("model_id") or routing.payload.get(
                "adapter_model_name"
            )
            cost = float(routing.payload.get("estimated_cost_usd") or 0.0)
        model = model or payload.get("router_model_id") or payload.get("model")
        if not cost:
            cost = float(payload.get("router_estimated_cost_usd") or 0.0)
        task_spans.append(
            TaskSpan(
                task_id=task.id,
                role=task.role,
                adapter=task.adapter,
                model=model,
                status=str(task.status),
                start_iso=task.created_at,
                end_iso=task.completed_at or task.updated_at,
                cost_usd=cost,
                failure=failure_by_task.get(task.id),
            )
        )

    return JobTrace(
        job_id=job.id,
        goal=job.goal,
        status=str(job.status),
        start_iso=job.created_at,
        end_iso=job.completed_at,
        tasks=task_spans,
    )


def _iso_to_unix_nanos(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(parse_iso(value).timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return None


def record_job_trace(
    job: Job,
    tasks: list[Task],
    artifacts: list[Artifact],
    *,
    env: Optional[dict] = None,
) -> bool:
    """Emit an OTel trace for a finished job. No-op unless tracing is enabled.

    Returns True if a trace was exported, False otherwise. Never raises — a
    telemetry failure must not fail a swarm.
    """
    if not telemetry_enabled(env):
        return False
    trace = build_job_trace(job, tasks, artifacts)
    try:
        return _emit_with_otel(trace, env=env if env is not None else os.environ)
    except Exception:
        return False


def _emit_with_otel(trace: JobTrace, *, env: dict) -> bool:
    global _warned_missing_sdk
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.trace import SpanContext, set_span_in_context
    except ImportError:
        if not _warned_missing_sdk:
            import sys

            print(
                "puppetmaster: OTEL tracing requested but the SDK is not installed; "
                "run `pip install 'puppetmaster-ai[otel]'`. Continuing without telemetry.",
                file=sys.stderr,
            )
            _warned_missing_sdk = True
        return False

    service_name = env.get("OTEL_SERVICE_NAME") or "puppetmaster"
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if (env.get("OTEL_TRACES_EXPORTER") or "").strip().lower() == "console":
        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))

    tracer = provider.get_tracer("puppetmaster")

    job_start = _iso_to_unix_nanos(trace.start_iso)
    job_end = _iso_to_unix_nanos(trace.end_iso)
    session = tracer.start_span(
        "puppetmaster.job",
        start_time=job_start,
        attributes=trace.attributes(),
    )
    parent_ctx = set_span_in_context(session)
    for task in trace.tasks:
        t_start = _iso_to_unix_nanos(task.start_iso) or job_start
        t_end = _iso_to_unix_nanos(task.end_iso) or job_end
        child = tracer.start_span(
            "puppetmaster.task",
            context=parent_ctx,
            start_time=t_start,
            attributes=task.attributes(),
        )
        if task.failure or task.status == "failed":
            child.set_status(ot_trace.Status(ot_trace.StatusCode.ERROR, task.failure or "failed"))
        child.end(end_time=t_end)
    session.end(end_time=job_end)

    provider.force_flush()
    provider.shutdown()
    return True
