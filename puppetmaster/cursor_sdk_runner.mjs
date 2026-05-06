#!/usr/bin/env node
import { Agent, CursorAgentError } from "@cursor/sdk";

const input = JSON.parse(process.env.PUPPETMASTER_CURSOR_INPUT || "{}");
const apiKey = process.env.CURSOR_API_KEY;

if (!apiKey) {
  console.error("CURSOR_API_KEY is required");
  process.exit(1);
}

try {
  const result = await Agent.prompt(input.prompt, {
    apiKey,
    model: { id: input.model || "default" },
    local: { cwd: input.cwd || process.cwd(), settingSources: [] },
  });

  process.stdout.write(
    JSON.stringify({
      status: result.status,
      result: result.result || "",
    }),
  );
  process.exit(result.status === "finished" ? 0 : 2);
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

