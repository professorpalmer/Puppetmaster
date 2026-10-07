/**
 * Puppetmaster TypeScript client — true blocking await for the SDK path.
 *
 * Cursor's MCP transport can't hold a long synchronous call open, so the MCP
 * `puppetmaster_await_job` tool is a *bounded* long-poll. Outside that stdio
 * constraint (a Node script, a CI step, a backend service) you can block for
 * real. This client does exactly that by driving Puppetmaster's durable CLI
 * (`python -m puppetmaster await <job_id> --json`), which talks to the same
 * SQLite/file-backed state the daemon writes — so it works from any process,
 * on any machine that shares the state dir, with zero new transport.
 *
 * Zero runtime dependencies (uses `node:child_process`). Ships as source; build
 * with your own tsc/bundler or import directly under a TS-aware runtime.
 *
 *   import { awaitJob } from "./puppetmaster";
 *   const result = await awaitJob("job_abc123", { timeoutSeconds: 0 });
 *   if (result.status === "complete") console.log(result.summary);
 */
import { spawn } from "node:child_process";

export interface AwaitJobResult {
  job_ref?: JobRef;
  job_id: string;
  status:
    | "complete"
    | "failed"
    | "stalled"
    | "cancelled"
    | "running"
    | "stitching"
    | "queued"
    | string;
  terminal: boolean;
  timed_out: boolean;
  completed_at: string | null;
  summary: string;
  delivery?: {
    verdict: "pending" | "delivered" | "degraded" | "blocked" | string;
    successful: boolean;
    status: string;
    quality: string | null;
    stale_task_ids: string[];
    incomplete_tasks: boolean;
    required_artifacts: boolean;
  };
  progress?: {
    last_substantive_artifact_at: string | null;
    last_liveness_at: string | null;
    last_substantive_artifact_age_seconds: number | null;
    last_liveness_age_seconds: number | null;
  };
}

export interface AwaitJobOptions {
  /** Seconds to wait before giving up. 0 (default) blocks until the job ends. */
  timeoutSeconds?: number;
  /** How often the CLI re-checks job state while blocked. Default 0.25s. */
  pollIntervalSeconds?: number;
  /** Python executable. Default "python3". */
  python?: string;
  /** Working directory (defaults to the current one). */
  cwd?: string;
  /** Extra env vars (merged over process.env), e.g. PUPPETMASTER_STATE_DIR. */
  env?: Record<string, string>;
  /**
   * Hard cap on how long this client itself will wait for the child process,
   * independent of the CLI's own --timeout-seconds. Defaults to no cap when
   * timeoutSeconds is 0, else timeoutSeconds + 30s of slack.
   */
  killAfterSeconds?: number;
}

export class PuppetmasterError extends Error {
  constructor(
    message: string,
    public readonly exitCode: number | null,
    public readonly stderr: string,
  ) {
    super(message);
    this.name = "PuppetmasterError";
  }
}

/**
 * Block until `jobId` reaches a terminal state (or the optional timeout), then
 * resolve with the job's final state + stitched summary. Rejects with a
 * {@link PuppetmasterError} if the CLI exits non-zero for a reason other than a
 * cleanly reported unsuccessful delivery (failed, stalled, cancelled, blocked,
 * empty, or degraded).
 */
export function awaitJob(
  jobId: string | JobRef,
  options: AwaitJobOptions = {},
): Promise<AwaitJobResult> {
  const {
    timeoutSeconds = 0,
    pollIntervalSeconds = 0.25,
    python = "python3",
    cwd,
    env,
    killAfterSeconds,
  } = options;

  const args = [
    "-m",
    "puppetmaster",
    ...(typeof jobId === "string" ? [] : ["--job-ref", JSON.stringify(jobId)]),
    "await",
    typeof jobId === "string" ? jobId : jobId.job_id,
    "--json",
    "--timeout-seconds",
    String(timeoutSeconds),
    "--poll-interval-seconds",
    String(pollIntervalSeconds),
  ];

  return new Promise<AwaitJobResult>((resolve, reject) => {
    const child = spawn(python, args, {
      cwd,
      env: { ...process.env, ...(env ?? {}) },
    });

    let stdout = "";
    let stderr = "";
    let timer: ReturnType<typeof setTimeout> | undefined;

    const cap =
      killAfterSeconds ?? (timeoutSeconds > 0 ? timeoutSeconds + 30 : undefined);
    if (cap !== undefined) {
      timer = setTimeout(() => {
        child.kill("SIGTERM");
      }, cap * 1000);
    }

    child.stdout.on("data", (chunk: unknown) => (stdout += String(chunk)));
    child.stderr.on("data", (chunk: unknown) => (stderr += String(chunk)));

    child.on("error", (err: Error) => {
      if (timer) clearTimeout(timer);
      reject(
        new PuppetmasterError(
          `failed to spawn ${python}: ${err.message}`,
          null,
          stderr,
        ),
      );
    });

    child.on("close", (code: number | null) => {
      if (timer) clearTimeout(timer);
      let parsed: AwaitJobResult | undefined;
      try {
        parsed = JSON.parse(stdout) as AwaitJobResult;
      } catch {
        parsed = undefined;
      }
      // `await` exits 1 for a parsed unsuccessful delivery. That is a successful
      // observation of a non-successful job, not a transport/client error.
      if (parsed && (code === 0 || code === 1)) {
        resolve(parsed);
        return;
      }
      reject(
        new PuppetmasterError(
          `puppetmaster await exited ${code} without parseable JSON`,
          code,
          stderr || stdout,
        ),
      );
    });
  });
}

/** Convenience: true once the job reached a terminal state (not timed out). */
export async function isJobDone(
  jobId: string | JobRef,
  options: AwaitJobOptions = {},
): Promise<boolean> {
  const result = await awaitJob(jobId, { ...options, timeoutSeconds: 0.001 });
  return result.terminal;
}

/** Store-scoped identity. Equal job IDs in separate stores are distinct. */
/** Read-only legacy selection. Mutation/replay requires explicit v2 rebinding. */
export interface MetadataClientOptions {
  readonly python?: string;
  readonly cwd?: string;
  readonly env?: Record<string, string>;
  readonly stateDir: string;
  readonly backend: "file" | "sqlite";
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function integer(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
function nullableText(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}
function isJobRef(value: unknown): value is JobRef {
  if (!record(value) || typeof value.job_id !== "string" || typeof value.state_id !== "string") return false;
  if (value.version === 2) return typeof value.incarnation === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value.incarnation);
  return (value.version === undefined || value.version === 1) && value.incarnation === undefined;
}
type PreviousMembershipFields = "previous_membership" | "previous_status" | "previous_origin" | "previous_project_id" | "previous_session_id";
type MetadataWireRef = Omit<MetadataRef, PreviousMembershipFields> & Partial<Pick<MetadataRef, PreviousMembershipFields>>;

function isMetadataWireRef(value: unknown): value is MetadataWireRef {
  if (!record(value) || !isJobRef(value.job_ref) || typeof value.id !== "string" ||
      !["job", "task", "artifact"].some(kind => kind === value.kind) || !integer(value.revision) ||
      !["known", "legacy_unknown"].some(stamp => stamp === value.stamp) || typeof value.deleted !== "boolean") return false;
  if (!["status", "sha256", "task_id", "artifact_type", "origin", "project_id", "session_id"].every(k => nullableText(value[k]))) return false;
  if (!["task_count", "artifact_count"].every(k => value[k] === null || integer(value[k]))) return false;
  if (!["previous_status", "previous_origin", "previous_project_id", "previous_session_id"].every(k => value[k] === undefined || nullableText(value[k]))) return false;
  if (value.previous_membership !== undefined && !["present", "absent", "unavailable"].some(membership => membership === value.previous_membership)) return false;
  const b = value.binding;
  if (b !== null && (!record(b) || typeof b.task_id !== "string" || (b.generation !== null && !integer(b.generation)) || !nullableText(b.lease_id) || !nullableText(b.owner))) return false;
  const hasPreview = Object.prototype.hasOwnProperty.call(value, "goal_preview");
  const hasTruncated = Object.prototype.hasOwnProperty.call(value, "goal_preview_truncated");
  if (hasPreview !== hasTruncated) return false;
  if (hasPreview) {
    if (!nullableText(value.goal_preview)) return false;
    if ((value.goal_preview === null) !== (value.goal_preview_truncated === null)) return false;
    if (value.goal_preview_truncated !== null && typeof value.goal_preview_truncated !== "boolean") return false;
    if (typeof value.goal_preview === "string" && Buffer.byteLength(value.goal_preview, "utf8") > 512) return false;
  }
  if (value.delivery !== undefined && !["pending", "blocked", "unverified", "unavailable"].some(delivery => delivery === value.delivery)) return false;
  return value.quality === undefined || value.quality === "unverified" || value.quality === "unavailable";
}
export function decodeMetadataPage(value: unknown): MetadataPage {
  if (!record(value) || !Array.isArray(value.items) || !value.items.every(isMetadataWireRef) || value.items.length > 200 ||
      !integer(value.revision) || !integer(value.scanned) || value.scanned > 1000 ||
      !nullableText(value.next_cursor) || !(value.reason === undefined || nullableText(value.reason)) ||
      !(value.retry_after_ms === undefined || value.retry_after_ms === null || integer(value.retry_after_ms))) throw new Error("Invalid metadata page");
  const outcome = value.outcome;
  if (outcome !== "complete" && outcome !== "partial" && outcome !== "unavailable" && outcome !== "cursor_expired") throw new Error("Invalid metadata outcome");
  return { items: value.items.map(item => ({...item,
      previous_status: item.previous_status ?? null, previous_origin: item.previous_origin ?? null,
      previous_project_id: item.previous_project_id ?? null, previous_session_id: item.previous_session_id ?? null,
      previous_membership: item.previous_membership ?? "unavailable", goal_preview: item.goal_preview ?? null,
      goal_preview_truncated: item.goal_preview_truncated ?? null, delivery: item.delivery ?? "unavailable",
      quality: item.quality ?? "unavailable"})), outcome, revision: value.revision, scanned: value.scanned,
      next_cursor: value.next_cursor, reason: value.reason ?? null, retry_after_ms: value.retry_after_ms ?? null };
}
function isSelectedMetric(value: unknown, count: number, money: boolean, equivalent: boolean): value is SelectedMetric {
  if (!record(value)) return false;
  const {known_selected: k, unknown_selected: u, estimated_selected: e, conflicting_selected: c, total, state} = value;
  if (!integer(k) || !integer(u) || !integer(e) || !integer(c) || k + u !== count || e > k || c > u) return false;
  const expected = !k ? "unknown" : u ? "partial" : e ? "estimated" : "measured";
  if (state !== expected || (equivalent && k !== e)) return false;
  if (u || !k) return total === null;
  return typeof total === "number" && Number.isFinite(total) && total >= 0 &&
    (money ? total <= 1e12 : Number.isSafeInteger(total));
}
function selectedTotals(value: unknown, count: number): SelectedTotals {
  if (!record(value) || !isSelectedMetric(value.tokens_in, count, false, false) ||
      !isSelectedMetric(value.tokens_out, count, false, false) ||
      !isSelectedMetric(value.cache_read_tokens, count, false, false) ||
      !isSelectedMetric(value.cache_write_tokens, count, false, false) ||
      !isSelectedMetric(value.api_cost_usd, count, true, false) ||
      !isSelectedMetric(value.plan_marginal_cost_usd, count, true, false) ||
      !isSelectedMetric(value.api_equivalent_cost_usd, count, true, true)) throw new Error("Invalid selected totals");
  return {tokens_in: value.tokens_in, tokens_out: value.tokens_out, cache_read_tokens: value.cache_read_tokens,
    cache_write_tokens: value.cache_write_tokens, api_cost_usd: value.api_cost_usd,
    plan_marginal_cost_usd: value.plan_marginal_cost_usd, api_equivalent_cost_usd: value.api_equivalent_cost_usd};
}
export function decodeSelectedEconomics(value: unknown, jobRef: JobRefV2): SelectedEconomics {
  if (!record(value)) throw new Error("Invalid selected economics");
  if (value.summary_revision === undefined && value.totals === undefined) return {
    job_ref: jobRef, outcome: "unavailable", summary_revision: null, receipt_digest: null,
    source: "unavailable", coverage: "unknown", selected_count: null, totals: null,
    reason: "projection_missing", retry_after_ms: null};
  if (!isJobRef(value.job_ref) || value.job_ref.version !== 2 || value.job_ref.job_id !== jobRef.job_id ||
      value.job_ref.state_id !== jobRef.state_id || value.job_ref.incarnation !== jobRef.incarnation ||
      !(value.summary_revision === null || integer(value.summary_revision)) ||
      !nullableText(value.receipt_digest) || (typeof value.receipt_digest === "string" && !/^[0-9a-f]{64}$/.test(value.receipt_digest)) ||
      !(value.selected_count === null || integer(value.selected_count)) ||
      !(value.retry_after_ms === null || integer(value.retry_after_ms))) throw new Error("Invalid selected identity or scalars");
  const {outcome, source, coverage, reason} = value;
  if (outcome !== "available" && outcome !== "unavailable") throw new Error("Invalid selected outcome");
  if (source !== "terminal_receipt" && source !== "unavailable") throw new Error("Invalid selected source");
  if (coverage !== "selected_receipt" && coverage !== "unknown") throw new Error("Invalid selected coverage");
  if (reason !== null && reason !== "no_terminal_receipt" && reason !== "legacy_provenance_unknown" && reason !== "projection_missing" &&
      reason !== "projection_pending" && reason !== "selection_changed" && reason !== "metadata_invalid" &&
      reason !== "numeric_limit" && reason !== "read_snapshot_unavailable") throw new Error("Invalid selected reason");
  if (outcome === "available" && (source !== "terminal_receipt" || coverage !== "selected_receipt" || reason !== null ||
      value.receipt_digest === null || value.selected_count === null || value.selected_count === 0)) throw new Error("Inconsistent selected provenance");
  if (outcome === "unavailable" && value.selected_count !== null) throw new Error("Invalid unavailable cardinality");
  const totals = value.totals === null ? null : selectedTotals(value.totals, value.selected_count ?? 0);
  if ((outcome === "available") !== (totals !== null)) throw new Error("Inconsistent selected outcome");
  return {job_ref: value.job_ref, outcome, source, coverage, reason, totals, summary_revision: value.summary_revision,
    receipt_digest: value.receipt_digest, selected_count: value.selected_count, retry_after_ms: value.retry_after_ms};
}
function boundedCommand(command: string[], options: MetadataClientOptions, bound: number, ref?: JobRef): Promise<unknown> {
  const args = ["-m", "puppetmaster", "--state-dir", options.stateDir, "--backend", options.backend,
    ...(ref ? ["--job-ref", JSON.stringify(ref)] : []), ...command, "--json"];
  return new Promise((resolve, reject) => {
    const child = spawn(options.python ?? "python3", args, {cwd: options.cwd, env: {...process.env, ...options.env}});
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    const stdout: string[] = [];
    const stderr: string[] = [];
    let outBytes = 0, errBytes = 0;
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new Error("Metadata command timed out")); }, 15000);
    child.stdout.on("data", (data: string) => {
      outBytes += Buffer.byteLength(data, "utf8");
      if (outBytes > bound + 1) { child.kill("SIGKILL"); reject(new Error("Metadata response exceeds byte bound")); }
      else stdout.push(data);
    });
    child.stderr.on("data", (data: string) => {
      errBytes += Buffer.byteLength(data, "utf8");
      if (errBytes > 8192) { child.kill("SIGKILL"); reject(new Error("Metadata stderr exceeds byte bound")); }
      else stderr.push(data);
    });
    child.on("error", error => {clearTimeout(timer); reject(error);});
    child.on("close", code => {
      clearTimeout(timer);
      if (code !== 0) return reject(new PuppetmasterError("Metadata command failed", code, stderr.join("")));
      try { const value: unknown = JSON.parse(stdout.join("")); resolve(value); }
      catch (error) { reject(error); }
    });
  });
}
function summaryArgs(command: string, options: JobSummaryOptions & {readonly after_revision?: number}): string[] {
  const args = [command];
  for (const name of ["cursor", "limit", "max_scan", "max_bytes", "status", "origin", "project_id", "session_id", "after_revision"] as const) {
    const value = options[name];
    if (value !== undefined && value !== null) args.push("--" + name.replaceAll("_", "-"), String(value));
  }
  return args;
}
export async function listJobSummaries(client: MetadataClientOptions, options: JobSummaryOptions = {}): Promise<MetadataPage> {
  return decodeMetadataPage(await boundedCommand(summaryArgs("job-summaries", options), client, options.max_bytes ?? 262144, options.job_ref));
}
export async function readJobSummaryChanges(client: MetadataClientOptions, options: JobSummaryOptions & {readonly after_revision?: number} = {}): Promise<MetadataPage> {
  return decodeMetadataPage(await boundedCommand(summaryArgs("job-summary-changes", options), client, options.max_bytes ?? 262144, options.job_ref));
}
export async function getSelectedEconomics(jobRef: JobRefV2, client: MetadataClientOptions,
  expectedSummaryRevision?: number): Promise<SelectedEconomics> {
  if (jobRef.version !== 2 || (expectedSummaryRevision !== undefined && !integer(expectedSummaryRevision))) throw new Error("Invalid selected request");
  const args = ["selected-economics", ...(expectedSummaryRevision === undefined ? [] : ["--expected-summary-revision", String(expectedSummaryRevision)])];
  return decodeSelectedEconomics(await boundedCommand(args, client, 8192, jobRef), jobRef);
}

export interface LegacyJobRef {
  readonly job_id: string;
  readonly state_id: string;
  readonly version?: 1;
  readonly incarnation?: never;
}

export interface IncarnatedJobRef {
  readonly job_id: string;
  readonly state_id: string;
  readonly version: 2;
  readonly incarnation: string;
}

export type JobRef = LegacyJobRef | IncarnatedJobRef;

export interface TaskBinding {
  readonly task_id: string;
  /** null denotes a legacy task with no known generation. */
  readonly generation: number | null;
  readonly lease_id: string | null;
  readonly owner: string | null;
}

export interface CompletionReceipt {
  readonly job_ref: JobRef;
  readonly run_id: string;
  readonly intent_digest: string | null;
  readonly outcome: "pending_publication" | "published" | "stale_lease" | "invalidated" | "legacy_unknown" | "unavailable";
}

export interface CancellationReceipt {
  readonly job_ref: JobRef;
  readonly request_id: string;
  readonly bindings: readonly TaskBinding[];
  readonly outcome: "requested" | "observed_stop" | "stale_binding" | "already_terminal" | "conflict";
  readonly revision: number;
  /** Local cleanup evidence only; never proof of stopped remote effects. */
  readonly cleanup: "unknown" | "partial" | "local_process_exited";
}

export interface EffectReceipt {
  readonly job_ref: JobRef;
  readonly effect_id: string;
  readonly request_digest: string;
  readonly binding: TaskBinding;
  readonly run_id: string;
  readonly attempt_id: string;
  readonly revision: number;
  readonly outcome: "not_dispatched" | "in_flight" | "succeeded" | "failed_no_effect" | "unknown";
  readonly replay_policy: "safe" | "reconcile_first" | "requires_authorization" | "provider_idempotent";
  readonly evidence_refs: readonly string[];
}

export interface MetadataRef {
  readonly job_ref: JobRef;
  readonly id: string;
  readonly kind: "job" | "task" | "artifact";
  readonly status: string | null;
  readonly sha256: string | null;
  readonly revision: number;
  readonly stamp: "known" | "legacy_unknown";
  readonly deleted: boolean;
  readonly task_count: number | null;
  readonly artifact_count: number | null;
  readonly binding: TaskBinding | null;
  readonly task_id: string | null;
  readonly artifact_type: string | null;
  readonly origin: string | null;
  readonly project_id: string | null;
  readonly session_id: string | null;
  readonly previous_membership: "present" | "absent" | "unavailable";
  readonly previous_status: string | null;
  readonly previous_origin: string | null;
  readonly previous_project_id: string | null;
  readonly previous_session_id: string | null;
  readonly goal_preview?: string | null;
  readonly goal_preview_truncated?: boolean | null;
  readonly delivery?: "pending" | "blocked" | "unverified" | "unavailable";
  readonly quality?: "unverified" | "unavailable";

}

/** Optional stamps on a job launch. Omitted legacy values remain unknown. */
export interface JobScope {
  readonly origin?: string | null;
  readonly project_id?: string | null;
  readonly session_id?: string | null;
}

export interface JobSummaryFilter extends JobScope {
  readonly status?: string;
  readonly job_ref?: JobRef;
}

export interface JobSummaryOptions extends MetadataPageOptions, JobSummaryFilter {}

export interface MetadataPage {
  readonly items: readonly MetadataRef[];
  readonly outcome: "complete" | "partial" | "unavailable" | "cursor_expired";
  readonly revision: number;
  readonly next_cursor: string | null;
  readonly scanned: number;
  readonly reason: string | null;
  readonly retry_after_ms: number | null;
}

/** Tokens bind query filters and store identity. Snapshot membership survives later writes.
 * Fetch one bounded page per consumer tick; continuation is not a drain request.
 * Projected scalar input is capped before Python hydration; oversized identities are unavailable. */
export interface MetadataPageOptions {
  readonly cursor?: string;
  readonly limit?: number; // 1..200
  readonly max_bytes?: number; // 1024..262144, including the JSON envelope
  readonly max_scan?: number; // 1..1000
  readonly status?: string;
}

export interface ConsumptionMetric {
  readonly total: number | null;
  readonly known_subtotal: number;
  readonly status: "unknown" | "partial" | "measured" | "estimated";
  readonly known_attempts: number;
  readonly unknown_attempts: number;
  readonly estimated_attempts: number;
  readonly conflicting_attempts: number;
}

export type ConsumptionTotals = Readonly<Record<
  "tokens_in" | "tokens_out" | "cache_read_tokens" | "cache_write_tokens" |
  "api_cost_usd" | "plan_marginal_cost_usd" | "api_equivalent_cost_usd", ConsumptionMetric>>;

export interface ExecutionAttempt {
  readonly job_id: string;
  readonly task_id: string;
  readonly run_id: string;
  readonly attempt_id: string;
  readonly started_at: string;
  readonly adapter: string;
  readonly model: string | null;
  readonly provider: string | null;
}

export interface ProcessOutcomeObservation {
  readonly job_id: string;
  readonly attempt_id: string;
  readonly observation_id: string;
  readonly source: string;
  readonly observed_at: string;
  readonly usage_state: "unknown" | "measured" | "estimated";
  readonly tokens_in: number | null;
  readonly tokens_out: number | null;
  readonly cache_read_tokens: number | null;
  readonly cache_write_tokens: number | null;
  readonly cost_state: "unknown" | "measured" | "estimated";
  readonly cost_usd: number | null;
  readonly cost_basis: "unknown" | "api" | "plan_marginal" | "api_equivalent";
  readonly returncode?: number | null;
  readonly timed_out?: boolean | null;
  /** Source data-quality flags (partial:/disputed:/conflict:/unlinked:); absent when none. */
  readonly quality?: readonly string[];
}

export interface AttemptConsumptionReport {
  readonly telemetry_coverage?: TelemetryCoverage;
  readonly complete_invocation_history?: false;
  readonly job_id: string;
  readonly attempt_count: number;
  readonly attempts: readonly {
    readonly attempt: ExecutionAttempt;
    readonly observation_ids: readonly string[];
    readonly process_outcomes: readonly ProcessOutcomeObservation[];
    readonly totals: ConsumptionTotals;
  }[];
  readonly totals: ConsumptionTotals;
}

/** Coverage of recorded telemetry; never proof of every provider invocation. */
export type TelemetryCoverage = "captured" | "partial" | "unknown";

export interface AttemptFacts {
  readonly job_id: string;
  readonly attempt_id: string;
  readonly task_id: string;
  readonly run_id: string;
  readonly started_at: string;
  readonly adapter: string;
  readonly model: string | null;
  readonly provider: string | null;
}

export interface RunFacts {
  readonly job_id: string;
  readonly id: string;
  readonly task_id: string;
  readonly role: string;
  readonly worker_id: string;
  readonly status: string;
  readonly started_at: string;
  readonly completed_at: string | null;
}

export interface ObservationFacts {
  readonly task_id: string | null;
  readonly run_id: string | null;
  readonly identity_state: "available" | "unavailable";
  readonly job_id: string;
  readonly attempt_id: string;
  readonly observation_id: string;
  readonly source: string;
  readonly observed_at: string;
  readonly usage_state: "unknown" | "measured" | "estimated";
  readonly tokens_in: number | null;
  readonly tokens_out: number | null;
  readonly cache_read_tokens: number | null;
  readonly cache_write_tokens: number | null;
  readonly cost_state: "unknown" | "measured" | "estimated";
  readonly cost_usd: number | null;
  readonly cost_basis: "unknown" | "api" | "plan_marginal" | "api_equivalent";
  readonly returncode: number | null;
  readonly timed_out: boolean | null;
  readonly quality?: readonly string[];
}

export type HistoricalRef = {
  readonly job_ref: JobRef;
  readonly sequence: number;
} & (
  | { readonly kind: "attempt"; readonly facts: AttemptFacts }
  | { readonly kind: "run"; readonly facts: RunFacts }
  | { readonly kind: "observation" | "outcome"; readonly facts: ObservationFacts }
);

export interface HistoricalPage {
  readonly items: readonly HistoricalRef[];
  readonly outcome: "complete" | "partial" | "unavailable" | "cursor_expired";
  readonly next_cursor: string | null;
  readonly scanned: number;
  readonly captured_count: number | null;
  readonly coverage: TelemetryCoverage;
  readonly complete_invocation_history: false;
}

export interface HistoricalCounts {
  readonly captured_attempts: number | null;
  readonly captured_runs: number | null;
  readonly captured_process_outcomes: number | null;
  readonly captured_observations: number | null;
  readonly outcome: "available" | "unavailable";
  readonly coverage: TelemetryCoverage;
  readonly complete_invocation_history: false;
}

export type JobSummary = MetadataRef & {readonly kind: "job"};
export type JobRefV2 = Extract<JobRef, {readonly version: 2}>;
export interface SelectedMetric {
  readonly total: number | null;
  readonly state: "unknown" | "partial" | "measured" | "estimated";
  readonly known_selected: number | null;
  readonly unknown_selected: number | null;
  readonly estimated_selected: number | null;
  readonly conflicting_selected: number | null;
}
export interface SelectedTotals {
  readonly tokens_in: SelectedMetric;
  readonly tokens_out: SelectedMetric;
  readonly cache_read_tokens: SelectedMetric;
  readonly cache_write_tokens: SelectedMetric;
  readonly api_cost_usd: SelectedMetric;
  readonly plan_marginal_cost_usd: SelectedMetric;
  readonly api_equivalent_cost_usd: SelectedMetric;
}
export interface SelectedEconomics {
  readonly job_ref: JobRefV2;
  readonly outcome: "available" | "unavailable";
  readonly summary_revision: number | null;
  readonly receipt_digest: string | null;
  readonly source: "terminal_receipt" | "unavailable";
  readonly coverage: "selected_receipt" | "unknown";
  readonly selected_count: number | null;
  readonly totals: SelectedTotals | null;
  readonly reason: "no_terminal_receipt" | "legacy_provenance_unknown" | "projection_missing" |
    "projection_pending" | "selection_changed" | "metadata_invalid" | "numeric_limit" | "read_snapshot_unavailable" | null;
  readonly retry_after_ms: number | null;
}
