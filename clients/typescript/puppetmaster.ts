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
  job_id: string;
  status: "complete" | "failed" | "running" | "stitching" | "queued" | string;
  terminal: boolean;
  timed_out: boolean;
  completed_at: string | null;
  summary: string;
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
 * cleanly-reported failed job.
 */
export function awaitJob(
  jobId: string,
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
    "await",
    jobId,
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
      // `await` exits 1 when the job itself FAILED but still prints valid JSON;
      // that's a successful await of a failed job, not a client error.
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
  jobId: string,
  options: AwaitJobOptions = {},
): Promise<boolean> {
  const result = await awaitJob(jobId, { ...options, timeoutSeconds: 0.001 });
  return result.terminal;
}
