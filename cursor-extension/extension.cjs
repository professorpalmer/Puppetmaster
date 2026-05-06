const vscode = require("vscode");
const cp = require("child_process");
const path = require("path");
const fs = require("fs");

const SECRET_CURSOR_API_KEY = "puppetmaster.cursorApiKey";
const SECRET_ANTHROPIC_API_KEY = "puppetmaster.anthropicApiKey";

let panelProvider;

function activate(context) {
  const output = vscode.window.createOutputChannel("Puppetmaster");
  panelProvider = new PuppetmasterPanelProvider(context, output);

  context.subscriptions.push(
    output,
    vscode.window.registerWebviewViewProvider("puppetmaster.controlPanel", panelProvider),
    vscode.commands.registerCommand("puppetmaster.openPanel", async () => {
      await vscode.commands.executeCommand("workbench.view.extension.puppetmaster");
    }),
    vscode.commands.registerCommand("puppetmaster.configureKeys", () => configureKeys(context)),
    vscode.commands.registerCommand("puppetmaster.doctor", () => runDoctor(context, output)),
    vscode.commands.registerCommand("puppetmaster.cursorReview", () => runCursor(context, output, "review")),
    vscode.commands.registerCommand("puppetmaster.cursorPlan", () => runCursor(context, output, "plan")),
    vscode.commands.registerCommand("puppetmaster.claudeImplement", () => runClaude(context, output)),
    vscode.commands.registerCommand("puppetmaster.showLast", () => showLast(context, output)),
    vscode.commands.registerCommand("puppetmaster.showLogs", () => showLogs(context, output)),
    vscode.commands.registerCommand("puppetmaster.showArtifacts", () => showArtifacts(context, output)),
  );
}

function deactivate() {}

class PuppetmasterPanelProvider {
  constructor(context, output) {
    this.context = context;
    this.output = output;
    this.view = undefined;
  }

  resolveWebviewView(webviewView) {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = renderPanelHtml();
    webviewView.webview.onDidReceiveMessage(async (message) => {
      if (!message || !message.command) {
        return;
      }
      await vscode.commands.executeCommand(message.command);
    });
    this.postStatus("Ready", "Open a workspace and run a swarm.");
  }

  postStatus(status, body) {
    if (!this.view) {
      return;
    }
    this.view.webview.postMessage({ type: "status", status, body });
  }

  postOutput(body) {
    if (!this.view) {
      return;
    }
    this.view.webview.postMessage({ type: "output", body });
  }
}

async function configureKeys(context) {
  const cursorKey = await vscode.window.showInputBox({
    title: "Cursor API Key",
    prompt: "Optional. Stored in Cursor secret storage.",
    password: true,
    ignoreFocusOut: true,
    placeHolder: "Leave blank to keep existing value",
  });
  if (cursorKey) {
    await context.secrets.store(SECRET_CURSOR_API_KEY, cursorKey);
  }

  const anthropicKey = await vscode.window.showInputBox({
    title: "Anthropic API Key",
    prompt: "Optional. Stored in Cursor secret storage and passed to Claude Code.",
    password: true,
    ignoreFocusOut: true,
    placeHolder: "Leave blank to keep existing value",
  });
  if (anthropicKey) {
    await context.secrets.store(SECRET_ANTHROPIC_API_KEY, anthropicKey);
  }

  vscode.window.showInformationMessage("Puppetmaster provider keys updated.");
}

async function runDoctor(context, output) {
  await runAndDisplay(context, output, ["doctor"], "Doctor");
}

async function runCursor(context, output, mode) {
  const prompt = await vscode.window.showInputBox({
    title: mode === "review" ? "Puppetmaster Cursor Review" : "Puppetmaster Cursor Plan",
    prompt: "What should the Cursor worker inspect?",
    value: mode === "review"
      ? "Review this repo and identify the highest leverage next step."
      : "Plan the next safe implementation slice for this repo.",
    ignoreFocusOut: true,
  });
  if (!prompt) {
    return;
  }
  const args = ["cursor", prompt, "--cwd", workspaceRoot(), "--dry-run"];
  if (mode === "review") {
    args.push("--review");
  }
  if (mode === "plan") {
    args.push("--plan");
  }
  await runAndDisplay(context, output, args, mode === "review" ? "Cursor Review" : "Cursor Plan");
}

async function runClaude(context, output) {
  const root = workspaceRoot();
  const status = await gitStatus(root);
  if (status.trim()) {
    const choice = await vscode.window.showWarningMessage(
      "Claude Code full-edit runs require a clean working tree by default. Create/use a worktree or explicitly allow dirty state?",
      "Cancel",
      "Run With --allow-dirty",
    );
    if (choice !== "Run With --allow-dirty") {
      return;
    }
  }

  const prompt = await vscode.window.showInputBox({
    title: "Puppetmaster Claude Code Implement",
    prompt: "What should Claude Code implement?",
    value: "Implement the approved change and run focused tests.",
    ignoreFocusOut: true,
  });
  if (!prompt) {
    return;
  }

  const permissionMode = configuration().get("defaultClaudePermissionMode", "acceptEdits");
  const args = [
    "claude",
    prompt,
    "--cwd",
    root,
    "--permission-mode",
    permissionMode,
  ];
  if (status.trim()) {
    args.push("--allow-dirty");
  }
  await runAndDisplay(context, output, args, "Claude Code Implement");
}

async function showLast(context, output) {
  await runAndDisplay(context, output, ["last"], "Last Job");
}

async function showLogs(context, output) {
  await runAndDisplay(context, output, ["logs"], "Logs");
}

async function showArtifacts(context, output) {
  const jobId = await latestJobId(context);
  if (!jobId) {
    vscode.window.showWarningMessage("No Puppetmaster jobs found.");
    return;
  }
  await runAndDisplay(context, output, ["artifacts", jobId], "Artifacts");
}

async function latestJobId(context) {
  const result = await runPuppetmaster(context, ["last"], { quiet: true });
  if (result.code !== 0) {
    return undefined;
  }
  return result.stdout.trim();
}

async function runAndDisplay(context, output, args, title) {
  panelProvider && panelProvider.postStatus(`Running ${title}`, `python -m puppetmaster ${args.join(" ")}`);
  output.show(true);
  output.appendLine(`\n# ${title}`);
  output.appendLine(`$ python -m puppetmaster ${args.map(shellQuote).join(" ")}`);

  try {
    const result = await runPuppetmaster(context, args);
    output.append(result.stdout);
    output.append(result.stderr);
    panelProvider && panelProvider.postOutput((result.stdout + result.stderr).trim() || "(no output)");
    panelProvider && panelProvider.postStatus(
      result.code === 0 ? `${title} complete` : `${title} failed`,
      `exit code ${result.code}`,
    );
    if (result.code !== 0) {
      vscode.window.showErrorMessage(`Puppetmaster ${title} failed with exit code ${result.code}.`);
      return;
    }
    vscode.window.showInformationMessage(`Puppetmaster ${title} complete.`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    output.appendLine(message);
    panelProvider && panelProvider.postStatus(`${title} failed`, message);
    vscode.window.showErrorMessage(message);
  }
}

async function runPuppetmaster(context, args, options = {}) {
  const root = workspaceRoot();
  const config = configuration();
  const pythonPath = config.get("pythonPath", "python");
  const stateDir = config.get("stateDir", ".puppetmaster");
  const commandArgs = ["-m", "puppetmaster", "--state-dir", stateDir, ...args];
  const env = await processEnvironment(context);

  return new Promise((resolve, reject) => {
    const child = cp.spawn(pythonPath, commandArgs, {
      cwd: root,
      env,
      shell: false,
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
      if (!options.quiet && panelProvider) {
        panelProvider.postOutput(stdout.trim());
      }
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => resolve({ code: code || 0, stdout, stderr }));
  });
}

async function processEnvironment(context) {
  const config = configuration();
  const env = { ...process.env };
  const cursorKey = await context.secrets.get(SECRET_CURSOR_API_KEY);
  const anthropicKey = await context.secrets.get(SECRET_ANTHROPIC_API_KEY);
  if (cursorKey) {
    env.CURSOR_API_KEY = cursorKey;
  }
  if (anthropicKey) {
    env.ANTHROPIC_API_KEY = anthropicKey;
  }
  env.CLAUDE_CODE_COMMAND = config.get("claudeCodeCommand", "npx -y @anthropic-ai/claude-code");
  return env;
}

function configuration() {
  return vscode.workspace.getConfiguration("puppetmaster");
}

function workspaceRoot() {
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  if (!folder) {
    throw new Error("Open a workspace folder before running Puppetmaster.");
  }
  return folder.uri.fsPath;
}

function gitStatus(cwd) {
  return new Promise((resolve) => {
    cp.execFile("git", ["status", "--short"], { cwd }, (error, stdout) => {
      resolve(error ? "" : stdout);
    });
  });
}

function shellQuote(value) {
  if (/^[A-Za-z0-9_./:=@-]+$/.test(value)) {
    return value;
  }
  return JSON.stringify(value);
}

function renderPanelHtml() {
  const nonce = String(Date.now());
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); }
    h1 { font-size: 18px; margin: 0 0 4px; }
    p { color: var(--vscode-descriptionForeground); line-height: 1.4; }
    button { width: 100%; margin: 5px 0; padding: 8px; color: var(--vscode-button-foreground); background: var(--vscode-button-background); border: 0; border-radius: 3px; cursor: pointer; }
    button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
    pre { white-space: pre-wrap; word-break: break-word; background: var(--vscode-editor-background); border: 1px solid var(--vscode-panel-border); padding: 8px; max-height: 280px; overflow: auto; }
    .status { border-left: 3px solid var(--vscode-focusBorder); padding-left: 8px; margin: 12px 0; }
  </style>
</head>
<body>
  <h1>Puppetmaster</h1>
  <p>Run independent Cursor and Claude Code workers through shared state, artifacts, leases, and memory.</p>
  <button data-command="puppetmaster.configureKeys">Configure Provider Keys</button>
  <button data-command="puppetmaster.doctor">Doctor</button>
  <button data-command="puppetmaster.cursorReview">Cursor Review Dry Run</button>
  <button data-command="puppetmaster.cursorPlan">Cursor Plan Dry Run</button>
  <button data-command="puppetmaster.claudeImplement">Claude Code Implement</button>
  <button class="secondary" data-command="puppetmaster.showLast">Show Last Job</button>
  <button class="secondary" data-command="puppetmaster.showLogs">Show Logs</button>
  <button class="secondary" data-command="puppetmaster.showArtifacts">Show Artifacts</button>
  <div class="status">
    <strong id="status">Starting</strong>
    <p id="body">Loading Puppetmaster...</p>
  </div>
  <pre id="output"></pre>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    for (const button of document.querySelectorAll("button[data-command]")) {
      button.addEventListener("click", () => vscode.postMessage({ command: button.dataset.command }));
    }
    window.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "status") {
        document.getElementById("status").textContent = message.status;
        document.getElementById("body").textContent = message.body || "";
      }
      if (message.type === "output") {
        document.getElementById("output").textContent = message.body || "";
      }
    });
  </script>
</body>
</html>`;
}

module.exports = { activate, deactivate };
