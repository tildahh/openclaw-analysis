import assert from "node:assert/strict";
import test from "node:test";
import { buildRecord, createObserver, redactText } from "./index.js";
import { configureLiveCapture, captureLiveRecord, captureModelOutput, buildSessionAttributes, buildRunAttributes, markAggregateUsage, addToolResultNames } from "./live-trace.js";

const context = { agentId: "minilda", sessionKey: "agent:minilda:main", sessionId: "chat-a", runId: "turn-a", trigger: "user" };

test("tool results retain their source names beside unchanged call IDs and content", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all" }, redactText);
  const messages = [{ role: "tool", parts: [{ type: "tool_call_response", id: "call-2", response: "second result" }] },
    { role: "tool", parts: [{ type: "tool_call_response", id: "call-1", response: "first result" }] }];
  const content = { inputMessages: [{ role: "toolResult", toolCallId: "call-1", toolName: "memory_search", content: "first result" },
    { role: "toolResult", toolCallId: "call-2", toolName: "web_search", content: "second result" }] };
  const original = structuredClone(content);
  const attrs = { "gen_ai.input.messages": JSON.stringify(messages), "input.value": JSON.stringify(messages) };
  addToolResultNames(attrs, context, content, { inputMessages: true });
  const named = JSON.parse(attrs["gen_ai.input.messages"]);
  assert.deepEqual(named.map((m) => m.name), ["web_search", "memory_search"]);
  assert.deepEqual(named.map((m) => m.parts), messages.map((m) => m.parts));
  assert.equal(attrs["input.value"], attrs["gen_ai.input.messages"]);
  assert.deepEqual(content, original);
});

test("tool labels respect capture scope and never guess an unknown or conflicting name", () => {
  const messages = [{ role: "tool", parts: [{ type: "tool_call_response", id: "unknown", response: "result" }] },
    { role: "tool", name: "existing", parts: [{ type: "tool_call_response", id: "call-1", response: "result" }] }];
  const content = { inputMessages: [{ role: "toolResult", toolCallId: "call-1", toolName: "memory_get" }] };
  const attrs = { "gen_ai.input.messages": JSON.stringify(messages), "input.value": JSON.stringify(messages) };
  const original = structuredClone(attrs);
  configureLiveCapture({ liveExport: true, sessionScope: "all" }, redactText);
  addToolResultNames(attrs, context, content, { inputMessages: true });
  assert.deepEqual(attrs, original);
  const other = { ...context, sessionKey: "agent:other:main" };
  for (const [event, policy] of [[context, {}], [other, { inputMessages: true }]]) {
    addToolResultNames(attrs, event, content, policy);
    assert.deepEqual(attrs, original);
  }
});

test("ordinary chats require explicit all-session capture and other agents stay excluded", () => {
  assert.equal(buildRecord("llm_input", { prompt: "Hello" }, context), null);
  assert.equal(buildRecord("llm_input", { prompt: "Hello" }, context, undefined, { sessionScope: "all" }).event.prompt, "Hello");
  assert.equal(buildRecord("llm_input", {}, { ...context, agentId: "other" }, undefined, { sessionScope: "all" }), null);
});

test("live capture is opt-in and respects content policy", () => {
  configureLiveCapture({}, redactText);
  assert.deepEqual(buildSessionAttributes(context), {});
  configureLiveCapture({ liveExport: true, sessionScope: "all", captureReasoning: "text" }, redactText);
  captureLiveRecord(buildRecord("llm_input", { prompt: "Question" }, { ...context, runId: "policy-check" }, undefined, { sessionScope: "all" }));
  assert.equal(buildRunAttributes(context, {}).attrs["input.value"], undefined);
  assert.equal(captureModelOutput(context, { outputMessages: [{ role: "assistant", reasoning: "private", content: "Answer" }] }, {}).reasoningText, undefined);
});

test("session grouping joins turns, separates fresh heartbeats, and exports no raw routing identifiers", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all" }, redactText);
  const a = buildSessionAttributes(context);
  assert.equal(a["session.id"], buildSessionAttributes({ ...context, runId: "turn-b" })["session.id"]);
  const hb = { ...context, sessionKey: "agent:minilda:main:heartbeat", trigger: "heartbeat" };
  assert.notEqual(a["session.id"], buildSessionAttributes(hb)["session.id"]);
  assert.notEqual(buildSessionAttributes(hb)["session.id"], buildSessionAttributes({ ...hb, sessionId: "fresh" })["session.id"]);
  assert.doesNotMatch(JSON.stringify(a), /chat-a|agent:minilda/);
});

test("the root displays the current question and last answer without copying conversation history", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all", captureReasoning: "text" }, redactText);
  captureLiveRecord(buildRecord("llm_input", { prompt: "Current question", historyMessages: [{ role: "user", content: "Old question" }] }, context, undefined, { sessionScope: "all" }));
  const first = { outputMessages: [{ role: "assistant", content: [{ type: "thinking", thinking: "First thought" }, { type: "toolCall", name: "read", arguments: {} }] }] };
  const second = { outputMessages: [{ role: "assistant", content: [{ type: "thinking", thinking: "Second thought" }, { type: "text", text: "Final answer" }] }] };
  const original = structuredClone(second);
  assert.equal(captureModelOutput(context, first, { outputMessages: true }).reasoningText, "First thought");
  assert.equal(captureModelOutput(context, second, { outputMessages: true }).reasoningText, "Second thought");
  const result = buildRunAttributes(context, { inputMessages: true, outputMessages: true });
  assert.equal(result.attrs["input.value"], "Current question");
  assert.equal(result.attrs["output.value"], "Final answer");
  assert.equal(result.attrs["openinference.span.kind"], "AGENT");
  assert.equal(result.name, "Conversation");
  assert.equal(result.attrs["openclaw.presentation.delivery"], "unknown");
  assert.deepEqual(second, original);
  assert.equal(buildRunAttributes({ ...context, trigger: "heartbeat" }, {}).name, "Heartbeat");
});

test("missing final output is reported explicitly and cannot reuse an earlier answer", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all" }, redactText);
  const evt = { ...context, runId: "missing-output" };
  captureModelOutput(evt, { outputMessages: [{ role: "assistant", content: "Interim" }] }, { outputMessages: true });
  captureModelOutput(evt, undefined, { outputMessages: true });
  assert.equal(buildRunAttributes(evt, { outputMessages: true }).attrs["output.value"], "[No final response captured]");
});

test("reasoning text is optional, bounded, redacted and not inferred from an attempt hook", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all", captureReasoning: "size" }, redactText);
  const content = { outputMessages: [{ role: "assistant", reasoning_content: "secret=abc", content: "Done" }] };
  assert.equal(captureModelOutput(context, content, { outputMessages: true }).reasoningText, undefined);
  configureLiveCapture({ liveExport: true, sessionScope: "all", captureReasoning: "text" }, redactText);
  const result = captureModelOutput(context, content, { outputMessages: true });
  assert.match(result.reasoningText, /REDACTED/);
  assert.equal(result.attrs["openclaw.presentation.reasoning_chars"], 10);
  assert.equal(captureModelOutput({ ...context, sessionKey: "agent:other:main" }, content, { outputMessages: true }).reasoningText, undefined);
});

test("aggregate usage retains native counters without double-counting model tokens", () => {
  const attrs = { "llm.token_count.total": 20, "gen_ai.usage.input_tokens": 15, "openclaw.tokens.total": 20 };
  markAggregateUsage(attrs, context);
  assert.equal(attrs["llm.token_count.total"], undefined);
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["openclaw.tokens.total"], 20);
  assert.equal(attrs["openinference.span.kind"], "CHAIN");
});

test("ordinary captures use a live subdirectory while existing evaluation collectors stay unchanged", () => {
  const paths = [];
  const observe = createObserver("agent_end", "/capture", null, (path) => paths.push(path), { sessionScope: "all" });
  observe({ success: true }, context);
  observe({ success: true }, { ...context, sessionKey: "agent:minilda:phoenix-eval-case" });
  assert.deepEqual(paths, ["/capture/live", "/capture"]);
});

test("separate turns in the same chat never inherit one another's request or answer", () => {
  configureLiveCapture({ liveExport: true, sessionScope: "all" }, redactText);
  const a = { ...context, runId: "first-turn" };
  const b = { ...context, runId: "second-turn" };
  captureLiveRecord(buildRecord("llm_input", { prompt: "First question" }, a, undefined, { sessionScope: "all" }));
  captureModelOutput(a, { outputMessages: [{ role: "assistant", content: "First answer" }] }, { outputMessages: true });
  captureLiveRecord(buildRecord("llm_input", { prompt: "Second question" }, b, undefined, { sessionScope: "all" }));
  const result = buildRunAttributes(b, { inputMessages: true, outputMessages: true }).attrs;
  assert.equal(result["input.value"], "Second question");
  assert.equal(result["output.value"], "[No final response captured]");
});
