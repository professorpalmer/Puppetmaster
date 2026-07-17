#!/usr/bin/env node
import { Agent, Cursor, CursorAgentError } from "@cursor/sdk";
import { SqliteLocalAgentStore } from "@cursor/sdk/sqlite";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const input = JSON.parse(process.env.PUPPETMASTER_CURSOR_INPUT || "{}");
const apiKey = process.env.CURSOR_API_KEY;

if (!apiKey) {
  console.error("CURSOR_API_KEY is required");
  process.exit(1);
}

// Discovery mode: enumerate the models the authenticated Cursor plan exposes.
// Puppetmaster's router treats these as plan-billed (no marginal API spend)
// and validates routed model ids against this catalog before dispatch.
if (input.mode === "list-models") {
  try {
    const models = await Cursor.models.list({ apiKey });
    process.stdout.write(
      JSON.stringify({
        ok: true,
        models: (models || []).map((m) => ({
          id: m.id,
          displayName: m.displayName,
          description: m.description ?? null,
        })),
      }),
    );
    process.exit(0);
  } catch (error) {
    console.error(
      error instanceof Error ? error.stack || error.message : String(error),
    );
    process.exit(1);
  }
}

try {
  // Token usage is the only honest measure of consumption on plan-billed
  // Cursor (marginal $0). The SDK exposes real input/output/cache counts ONLY
  // on the streaming `turn-ended` delta — `Agent.prompt()`'s RunResult carries
  // no usage at all, which is why the old `result.usage` read was always null
  // and the Python side silently fell back to a wild char/4 undercount. So we
  // run via Agent.create + agent.send({ onDelta }) and sum per-turn usage
  // across the whole agentic run (an implement task spans many turns).
  // Each Puppetmaster worker is a separate process. Give each process its own
  // SDK SQLite root: the SDK's default root is workspace-scoped, so parallel
  // workers otherwise all open the same index.db and can fail with SQLITE_BUSY /
  // "database is locked" before the model run starts.
  const workspaceRef = input.cwd || process.cwd();
  const sdkStateRoot = await mkdtemp(join(tmpdir(), "puppetmaster-cursor-sdk-"));
  const store = await SqliteLocalAgentStore.open({
    workspaceRef,
    stateRoot: sdkStateRoot,
  });
  let agent;

  let exitCode = 1;
  let output = "";
  try {
    agent = await Agent.create({
      apiKey,
      model: { id: input.model || "default" },
      local: { cwd: workspaceRef, settingSources: [], store },
    });
    const usage = {
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    };
    let measured = false;
    const run = await agent.send(input.prompt, {
      onDelta: ({ update }) => {
        if (update && update.type === "turn-ended" && update.usage) {
          measured = true;
          usage.inputTokens += update.usage.inputTokens || 0;
          usage.outputTokens += update.usage.outputTokens || 0;
          usage.cacheReadTokens += update.usage.cacheReadTokens || 0;
          usage.cacheWriteTokens += update.usage.cacheWriteTokens || 0;
        }
      },
    });
    const result = await run.wait();
    // usage stays null when the runtime never reported it, so the Python side
    // falls back to a clearly-labeled estimate rather than a fake-measured 0.
    output = JSON.stringify({
      status: result.status,
      result: result.result || "",
      usage: measured ? usage : null,
      requestId: result.requestId ?? run.requestId ?? null,
    });
    exitCode = result.status === "finished" ? 0 : 2;
  } finally {
    // Always dispose before exiting — process.exit() would skip pending
    // finalizers, leaking the local executor + run watchers.
    if (agent && typeof agent[Symbol.asyncDispose] === "function") {
      await agent[Symbol.asyncDispose]();
    } else if (agent && typeof agent.dispose === "function") {
      await agent.dispose();
    }
    await store.dispose();
    await rm(sdkStateRoot, { recursive: true, force: true });
  }
  if (output) process.stdout.write(output);
  process.exit(exitCode);
} catch (error) {
  if (error instanceof CursorAgentError) {
    console.error(
      JSON.stringify({
        type: "CursorAgentError",
        message: error.message,
        retryable: error.isRetryable,
      }),
    );
    process.exit(1);
  }
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
}

