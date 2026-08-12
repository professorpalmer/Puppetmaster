/**
 * Minimal entry for the Grok Bot remote-MCP e2e fixture.
 * Exists so package.json `main` resolves and review/implement workers
 * have a real source file to touch — not a production API.
 */

export const FIXTURE_NAME = "puppetmaster-grok-bot-e2e";

export function hello(name = "Grok Bot") {
  return `hello from ${FIXTURE_NAME}, ${name}`;
}
