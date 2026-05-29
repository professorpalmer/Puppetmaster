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
import secrets
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


def live_telemetry_enabled(env: Optional[dict] = None) -> bool:
    """True when per-task spans should be emitted live, as each worker finishes.

    Opt-in on top of tracing via ``PUPPETMASTER_OTEL_LIVE``. When on, workers
    emit their own ``puppetmaster.task`` span at completion (correlated to the
    job via the propagated ``TRACEPARENT``), and the end-of-job emission writes
    only the root job span so task spans aren't duplicated. When off (default),
    the whole trace is assembled once at job completion — reliable, no
    cross-process span plumbing required.
    """
    env = env if env is not None else os.environ
    if not telemetry_enabled(env):
        return False
    return (env.get("PUPPETMASTER_OTEL_LIVE") or "").strip().lower() in _TRUE


# ----- W3C trace context (cross-process correlation) -----------------------


def new_traceparent() -> str:
    """Mint a fresh W3C ``traceparent`` (version-traceid-spanid-flags, sampled).

    The orchestrator generates one per job and exports it to worker
    subprocesses via the standard ``TRACEPARENT`` env var, so the job span and
    every (possibly cross-process) task span land in a single correlated trace.
    """
    trace_id = secrets.token_hex(16)  # 128-bit
    span_id = secrets.token_hex(8)  # 64-bit
    return f"00-{trace_id}-{span_id}-01"


def parse_traceparent(traceparent: Optional[str]) -> "Optional[tuple[int, int]]":
    """Parse a ``traceparent`` into ``(trace_id, span_id)`` ints, or None.

    Tolerant of malformed input (returns None) — telemetry never raises.
    """
    if not traceparent:
        return None
    parts = traceparent.strip().split("-")
    if len(parts) != 4:
        return None
    _version, trace_hex, span_hex, _flags = parts
    if len(trace_hex) != 32 or len(span_hex) != 16:
        return None
    try:
        trace_id = int(trace_hex, 16)
        span_id = int(span_hex, 16)
    except ValueError:
        return None
    if trace_id == 0 or span_id == 0:
        return None
    return trace_id, span_id


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


def build_task_span(task: Task, artifacts: list[Artifact]) -> TaskSpan:
    """Build the :class:`TaskSpan` for a single task (live per-worker emission).

    Mirrors the per-task assembly in :func:`build_job_trace` but scoped to one
    task, so a worker can emit its own span the moment it finishes."""
    task_artifacts = [a for a in artifacts if a.task_id == task.id]
    routing = next(
        (a for a in task_artifacts if a.type == ArtifactType.ROUTING), None
    )
    failure = None
    for artifact in task_artifacts:
        candidate = (artifact.payload or {}).get("failure")
        if candidate:
            failure = str(candidate)
            break
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
    return TaskSpan(
        task_id=task.id,
        role=task.role,
        adapter=task.adapter,
        model=model,
        status=str(task.status),
        start_iso=task.created_at,
        end_iso=task.completed_at or task.updated_at,
        cost_usd=cost,
        failure=failure,
    )


def build_job_metrics(trace: JobTrace) -> dict:
    """Pure, SDK-free metric snapshot for a finished job.

    Returns the counters/sums an OTel meter would record. Exposed separately so
    it is unit-testable without the metrics SDK."""
    by_status: dict[str, int] = {}
    by_failure: dict[str, int] = {}
    for span in trace.tasks:
        by_status[span.status] = by_status.get(span.status, 0) + 1
        if span.failure:
            by_failure[span.failure] = by_failure.get(span.failure, 0) + 1
    return {
        "jobs": 1,
        "job_status": trace.status,
        "tasks": len(trace.tasks),
        "tasks_by_status": by_status,
        "tasks_by_failure": by_failure,
        "cost_usd": trace.total_cost_usd,
    }


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
    traceparent: Optional[str] = None,
) -> bool:
    """Emit an OTel trace for a finished job. No-op unless tracing is enabled.

    When ``traceparent`` is provided, the job span attaches to that propagated
    trace context so it correlates with any live worker spans. In live mode the
    end-of-job emission writes only the root job span (task spans were already
    emitted by the workers); otherwise it emits the full job→task tree.

    Returns True if a trace was exported, False otherwise. Never raises — a
    telemetry failure must not fail a swarm.
    """
    env = env if env is not None else os.environ
    if not telemetry_enabled(env):
        return False
    trace = build_job_trace(job, tasks, artifacts)
    include_task_spans = not live_telemetry_enabled(env)
    try:
        return _emit_with_otel(
            trace,
            env=env,
            traceparent=traceparent,
            include_task_spans=include_task_spans,
        )
    except Exception:
        return False


def record_task_span(
    job_goal: str,
    task: Task,
    artifacts: list[Artifact],
    *,
    traceparent: Optional[str],
    env: Optional[dict] = None,
) -> bool:
    """Emit a single live ``puppetmaster.task`` span as a worker finishes.

    No-op unless live telemetry is enabled. Attaches to the propagated
    ``traceparent`` so the span lands in the job's trace even though the worker
    is a separate OS process. Never raises."""
    env = env if env is not None else os.environ
    if not live_telemetry_enabled(env):
        return False
    span = build_task_span(task, artifacts)
    try:
        return _emit_task_span_with_otel(span, env=env, traceparent=traceparent)
    except Exception:
        return False


def record_job_metrics(
    job: Job,
    tasks: list[Task],
    artifacts: list[Artifact],
    *,
    env: Optional[dict] = None,
) -> bool:
    """Record OTel metrics (job/task counters + cost) for a finished job.

    No-op unless tracing is enabled and the metrics SDK is installed. Never
    raises."""
    env = env if env is not None else os.environ
    if not telemetry_enabled(env):
        return False
    trace = build_job_trace(job, tasks, artifacts)
    metrics = build_job_metrics(trace)
    try:
        return _emit_metrics_with_otel(metrics, env=env)
    except Exception:
        return False


def _build_provider_and_parent(env: dict, traceparent: Optional[str]):
    """Construct a TracerProvider + optional propagated parent context.

    Returns ``(provider, tracer, parent_ctx)`` or raises ImportError when the
    SDK is missing (handled by callers)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        set_span_in_context,
    )

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

    parent_ctx = None
    parsed = parse_traceparent(traceparent)
    if parsed is not None:
        trace_id, span_id = parsed
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        parent_ctx = set_span_in_context(NonRecordingSpan(span_context))
    return provider, tracer, parent_ctx


def _emit_task_span_with_otel(
    span: TaskSpan, *, env: dict, traceparent: Optional[str]
) -> bool:
    global _warned_missing_sdk
    try:
        from opentelemetry import trace as ot_trace
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
    provider, tracer, parent_ctx = _build_provider_and_parent(env, traceparent)
    start = _iso_to_unix_nanos(span.start_iso)
    end = _iso_to_unix_nanos(span.end_iso)
    child = tracer.start_span(
        "puppetmaster.task",
        context=parent_ctx,
        start_time=start,
        attributes=span.attributes(),
    )
    if span.failure or span.status == "failed":
        child.set_status(
            ot_trace.Status(ot_trace.StatusCode.ERROR, span.failure or "failed")
        )
    child.end(end_time=end)
    provider.force_flush()
    provider.shutdown()
    return True


def _emit_metrics_with_otel(metrics: dict, *, env: dict) -> bool:
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        return False

    service_name = env.get("OTEL_SERVICE_NAME") or "puppetmaster"
    resource = Resource.create({"service.name": service_name})
    if (env.get("OTEL_METRICS_EXPORTER") or env.get("OTEL_TRACES_EXPORTER") or "").strip().lower() == "console":
        exporter = ConsoleMetricExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        exporter = OTLPMetricExporter()
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = provider.get_meter("puppetmaster")

    jobs = meter.create_counter("puppetmaster.jobs", description="Jobs completed")
    tasks_counter = meter.create_counter(
        "puppetmaster.tasks", description="Worker tasks executed"
    )
    cost = meter.create_counter(
        "puppetmaster.cost_usd", unit="USD", description="Estimated spend"
    )
    jobs.add(metrics["jobs"], {"job.status": metrics["job_status"]})
    for status, count in metrics["tasks_by_status"].items():
        tasks_counter.add(count, {"task.status": status})
    cost.add(metrics["cost_usd"], {"job.status": metrics["job_status"]})

    provider.force_flush()
    provider.shutdown()
    return True


def _emit_with_otel(
    trace: JobTrace,
    *,
    env: dict,
    traceparent: Optional[str] = None,
    include_task_spans: bool = True,
) -> bool:
    global _warned_missing_sdk
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.trace import set_span_in_context
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

    provider, tracer, propagated_parent = _build_provider_and_parent(env, traceparent)

    job_start = _iso_to_unix_nanos(trace.start_iso)
    job_end = _iso_to_unix_nanos(trace.end_iso)
    session = tracer.start_span(
        "puppetmaster.job",
        context=propagated_parent,
        start_time=job_start,
        attributes=trace.attributes(),
    )
    if include_task_spans:
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
