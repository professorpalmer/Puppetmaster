import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type, type TSchema } from "typebox";

type McpTool = {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
};

type JsonRpc = {
  jsonrpc: "2.0";
  id?: number;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { message?: string };
};

function agentDir(): string {
  return process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
}

function loadMcpEntry(): { command: string; args: string[]; env?: Record<string, string> } {
  const mcpPath = join(agentDir(), "mcp.json");
  if (existsSync(mcpPath)) {
    const raw = JSON.parse(readFileSync(mcpPath, "utf8"));
    const entry = raw?.mcpServers?.puppetmaster;
    if (entry?.command) {
      return {
        command: String(entry.command),
        args: Array.isArray(entry.args) ? entry.args.map(String) : ["-m", "puppetmaster.mcp_server"],
        env: entry.env && typeof entry.env === "object" ? entry.env : undefined,
      };
    }
  }
  const python = process.env.PUPPETMASTER_PI_PYTHON || process.env.PYTHON || "python3";
  return { command: python, args: ["-m", "puppetmaster.mcp_server"] };
}

function schemaToTypebox(schema: unknown): TSchema {
  if (!schema || typeof schema !== "object") return Type.Any();
  const s = schema as Record<string, unknown>;
  const t = s.type;
  if (t === "string") return Type.String({ description: String(s.description || "") });
  if (t === "integer") return Type.Integer({ description: String(s.description || "") });
  if (t === "number") return Type.Number({ description: String(s.description || "") });
  if (t === "boolean") return Type.Boolean({ description: String(s.description || "") });
  if (t === "array") return Type.Array(schemaToTypebox(s.items), { description: String(s.description || "") });
  const props = (s.properties || {}) as Record<string, unknown>;
  const required = new Set(Array.isArray(s.required) ? (s.required as string[]) : []);
  const out: Record<string, TSchema> = {};
  for (const [key, value] of Object.entries(props)) {
    const inner = schemaToTypebox(value);
    out[key] = required.has(key) ? inner : Type.Optional(inner);
  }
  return Type.Object(out, { additionalProperties: true });
}

class StdioMcp {
  private child: ChildProcessWithoutNullStreams | null = null;
  private buf = "";
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  start(): void {
    if (this.child) return;
    const spec = loadMcpEntry();
    this.child = spawn(spec.command, spec.args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...(spec.env || {}) },
    });
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => {
      this.buf += chunk;
      let idx: number;
      while ((idx = this.buf.indexOf("\n")) !== -1) {
        const line = this.buf.slice(0, idx).trim();
        this.buf = this.buf.slice(idx + 1);
        if (!line.startsWith("{")) continue;
        let msg: JsonRpc;
        try {
          msg = JSON.parse(line) as JsonRpc;
        } catch {
          continue;
        }
        if (msg.id == null) continue;
        const waiter = this.pending.get(Number(msg.id));
        if (!waiter) continue;
        this.pending.delete(Number(msg.id));
        if (msg.error) waiter.reject(new Error(msg.error.message || "mcp error"));
        else waiter.resolve(msg.result);
      }
    });
    this.child.on("exit", () => {
      this.child = null;
      for (const waiter of this.pending.values()) waiter.reject(new Error("mcp exited"));
      this.pending.clear();
    });
  }

  stop(): void {
    if (!this.child) return;
    this.child.kill();
    this.child = null;
  }

  request(method: string, params: unknown): Promise<unknown> {
    this.start();
    const id = this.nextId++;
    const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child!.stdin.write(payload, (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }
}

export default function puppetmasterPiPilot(pi: ExtensionAPI) {
  const mcp = new StdioMcp();
  let registered = false;

  async function registerTools() {
    if (registered) return;
    const listed = (await mcp.request("tools/list", {})) as { tools?: McpTool[] };
    const tools = listed?.tools || [];
    for (const tool of tools) {
      const name = tool.name;
      pi.registerTool({
        name,
        label: name.replace(/^puppetmaster_/, ""),
        description: tool.description || name,
        parameters: schemaToTypebox(tool.inputSchema || {}),
        promptSnippet: tool.description || name,
        async execute(_id, params, signal) {
          if (signal?.aborted) throw new Error("cancelled");
          const result = (await mcp.request("tools/call", { name, arguments: params })) as {
            content?: Array<{ type?: string; text?: string }>;
            isError?: boolean;
          };
          const text = (result?.content || [])
            .map((part) => part.text || "")
            .join("\n")
            .slice(0, 50_000);
          if (result?.isError) throw new Error(text || "mcp tool error");
          return { content: [{ type: "text", text: text || "{}" }], details: result };
        },
      });
    }
    registered = true;
  }

  pi.on("session_start", async () => {
    try {
      mcp.start();
      await registerTools();
    } catch (err) {
      registered = false;
      const message = err instanceof Error ? err.message : String(err);
      pi.registerTool({
        name: "puppetmaster_mcp_connect_error",
        label: "puppetmaster mcp error",
        description: "Puppetmaster MCP stdio failed to connect. Run puppetmaster install-pi-mcp.",
        parameters: Type.Object({}),
        async execute() {
          return { content: [{ type: "text", text: message }] };
        },
      });
    }
  });

  pi.on("session_shutdown", async () => {
    mcp.stop();
  });
}
