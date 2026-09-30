import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import plugin, { buildRecord, createObserver, extractReasoning, redactText, resolveScope, sanitizeValue, writeRecord } from "./index.js";

const CONTEXT = { agentId: "minilda", sessionKey: "agent:minilda:phoenix-eval-case1", sessionId: "session1", runId: "run1" };

test("scope rejects ordinary, empty, mismatched and incomplete sessions", () => {
  for (const context of [
    {}, { ...CONTEXT, sessionKey: "agent:minilda:main" },
    { ...CONTEXT, sessionKey: "agent:minilda:phoenix-eval-" },
    { ...CONTEXT, agentId: "other" }, { ...CONTEXT, runId: undefined },
    { ...CONTEXT, sessionKey: "agent:other:phoenix-eval-case1" },
  ]) assert.equal(resolveScope({}, context), null);
  assert.deepEqual(resolveScope({}, CONTEXT), { ...CONTEXT, trace: undefined });
});

test("model timing hooks scope from session identity when core omits agentId", () => {
  const event = { runId: "run1", sessionKey: CONTEXT.sessionKey, sessionId: "session1", callId: "call1" };
  assert.equal(resolveScope(event, {})?.agentId, "minilda");
  assert.equal(resolveScope(event, { agentId: "other" }), null);
});

test("visible history survives while system prompts, reasoning and images never appear", () => {
  const event = {
    prompt: "Please inspect the fixture.", systemPrompt: "PRIVATE SYSTEM PROMPT", imagesCount: 1,
    historyMessages: [
      { role: "system", content: "PRIVATE SYSTEM MESSAGE" },
      { role: "developer", content: "PRIVATE DEVELOPER MESSAGE" },
      { role: "user", content: "Fixture question" },
      { role: "assistant", content: [
        { type: "thinking", thinking: "PRIVATE THOUGHT" },
        { type: "reasoning", text: "PRIVATE REASONING" },
        { type: "text", text: "Visible reply" },
        { type: "image", data: "BINARY IMAGE" },
      ] },
    ],
  };
  const original = structuredClone(event);
  const record = buildRecord("llm_input", event, CONTEXT, "2026-09-28T00:00:00Z");
  const text = JSON.stringify(record);
  assert.equal(record.event.systemPromptChars, event.systemPrompt.length);
  assert.match(record.event.systemPromptSha256, /^[0-9a-f]{64}$/);
  assert.match(text, /Fixture question/);
  assert.match(text, /Visible reply/);
  assert.doesNotMatch(text, /PRIVATE|BINARY IMAGE/);
  assert.deepEqual(event, original);
});

test("explicit hook fields exclude raw assistant, full terminal messages and arbitrary extras", () => {
  const output = buildRecord("llm_output", { assistantTexts: ["Done"], usage: { input: 9, output: 3 }, lastAssistant: { thinking: "hidden" }, extra: "hidden" }, CONTEXT);
  assert.deepEqual(output.event, { assistantTexts: ["Done"], usage: { input: 9, output: 3 } });
  const terminal = buildRecord("agent_end", { success: true, durationMs: 12, messages: ["hidden"] }, CONTEXT);
  assert.deepEqual(terminal.event, { success: true, durationMs: 12 });
});

test("reasoning is omitted by default, sized on request, and captured as text only when configured", () => {
  const lastAssistant = { content: [
    { type: "thinking", thinking: "First thought about api_key=sk-abcdefghijklmnop." },
    { type: "reasoning", text: "Second thought." },
    { type: "text", text: "Visible reply" },
  ], reasoning_content: "Server-side reasoning." };
  const event = { assistantTexts: ["Visible reply"], usage: { input: 1, output: 1 }, lastAssistant };
  assert.deepEqual(extractReasoning(lastAssistant), { chars: 48 + 15 + 22, texts: ["Server-side reasoning.", "First thought about api_key=sk-abcdefghijklmnop.", "Second thought."] });
  assert.deepEqual(extractReasoning({ thinking: "plain" }), { chars: 5, texts: ["plain"] });
  assert.deepEqual(extractReasoning(undefined), { chars: 0, texts: [] });
  const none = buildRecord("llm_output", event, CONTEXT, "2026-09-28T00:00:00Z");
  assert.deepEqual(Object.keys(none.event).sort(), ["assistantTexts", "usage"]);
  const size = buildRecord("llm_output", event, CONTEXT, "2026-09-28T00:00:00Z", { captureReasoning: "size" });
  assert.equal(size.event.reasoningChars, 85);
  assert.equal(size.event.reasoningTexts, undefined);
  const text = buildRecord("llm_output", event, CONTEXT, "2026-09-28T00:00:00Z", { captureReasoning: "text" });
  assert.equal(text.event.reasoningTexts.length, 3);
  assert.match(JSON.stringify(text.event.reasoningTexts), /Second thought/);
  assert.doesNotMatch(JSON.stringify(text.event.reasoningTexts), /sk-abcdefghijklmnop/);
  assert.doesNotMatch(JSON.stringify(text.event), /lastAssistant/);
});

test("credential fields and common literals are redacted without removing useful results", () => {
  const cleaned = sanitizeValue({
    headers: { Authorization: "Bearer private", "x-api-key": "private", Cookie: "private" },
    env: { VLLM_API_KEY: "private", AWS_SECRET_ACCESS_KEY: "private" },
    params: { command: "API_KEY='private' run-command" },
    result: { data: { exitCode: 1, text: "File not found" } },
  });
  assert.doesNotMatch(JSON.stringify(cleaned), /private/);
  assert.equal(cleaned.result.data.exitCode, 1);
  assert.equal(cleaned.result.data.text, "File not found");
  assert.doesNotMatch(redactText("Bearer abc123 https://user:secret@example.com sk-abcdefghijklmno"), /abc123|user:secret|abcdefghijklmno/);
  assert.equal(sanitizeValue(Buffer.from("private")), undefined);
});

test("long text explicitly identifies omitted evidence", () => {
  assert.equal(redactText("a".repeat(16000)), "a".repeat(16000));
  assert.equal(redactText("a".repeat(16017)), "a".repeat(16000) + "\n[TRUNCATED: 17 additional characters omitted]");
});

test("scope guard runs before inspecting content and capture never changes a tool result", () => {
  const event = {};
  Object.defineProperty(event, "prompt", { get() { throw new Error("must not inspect"); } });
  assert.equal(buildRecord("llm_input", event, { sessionKey: "agent:minilda:main" }), null);
  const rows = [];
  const callback = createObserver("before_tool_call", "/unused", null, (_, row) => rows.push(row));
  const params = { path: "/tmp/fixture" };
  assert.equal(callback({ toolName: "read", params }, CONTEXT), undefined);
  assert.deepEqual(params, { path: "/tmp/fixture" });
  assert.equal(rows.length, 1);
});

test("I/O and logger failures cannot block a before-tool hook or leak error details", () => {
  const warnings = [];
  const callback = createObserver("before_tool_call", "/unused", { warn: (message) => warnings.push(message) }, () => { throw new Error("SECRET path and input"); });
  assert.equal(callback({ toolName: "read", params: {} }, CONTEXT), undefined);
  assert.deepEqual(warnings, ["phoenix-eval-observer: local capture failed for before_tool_call"]);
  const brokenLogger = createObserver("before_tool_call", "/unused", { warn() { throw Error("logger failed"); } }, () => { throw Error("I/O failed"); });
  assert.doesNotThrow(() => brokenLogger({ toolName: "read" }, CONTEXT));
});

test("records append privately and refuse a destination symlink", () => {
  const parent = mkdtempSync(join(tmpdir(), "phoenix-observer-"));
  const directory = join(parent, "records");
  try {
    const record = buildRecord("agent_end", { success: true }, CONTEXT);
    writeRecord(directory, record);
    writeRecord(directory, record);
    const filename = join(directory, readdirSync(directory)[0]);
    assert.equal(statSync(directory).mode & 0o777, 0o700);
    assert.equal(statSync(filename).mode & 0o777, 0o600);
    assert.equal(readFileSync(filename, "utf8").trim().split("\n").length, 2);
    const target = join(parent, "target");
    writeFileSync(target, "untouched");
    rmSync(filename);
    symlinkSync(target, filename);
    assert.throws(() => writeRecord(directory, record));
    assert.equal(readFileSync(target, "utf8"), "untouched");
  } finally { rmSync(parent, { recursive: true, force: true }); }
});

test("plugin registers exactly the seven passive capture callbacks", () => {
  const hooks = [];
  plugin.register({ pluginConfig: {}, on: (hook, callback) => hooks.push([hook, callback]) });
  assert.deepEqual(hooks.map(([hook]) => hook), ["llm_input", "llm_output", "before_tool_call", "after_tool_call", "model_call_started", "model_call_ended", "agent_end"]);
  for (const [, callback] of hooks) assert.equal(callback({}, {}), undefined);
});
