import assert from "node:assert/strict";
import test from "node:test";
import { FIXTURE_NAME, hello } from "./index.js";

test("fixture exports a stable name", () => {
  assert.equal(FIXTURE_NAME, "puppetmaster-grok-bot-e2e");
});

test("hello returns a greeting", () => {
  assert.match(hello("e2e"), /puppetmaster-grok-bot-e2e/);
  assert.match(hello("e2e"), /e2e/);
});
