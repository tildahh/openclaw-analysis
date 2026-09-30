import { createHash } from "node:crypto";

// The plugin loader and native exporter may import different module instances.
const state = globalThis[Symbol.for("openclaw.phoenix.live-trace.v1")] ??= { options: {}, runs: new Map(), clean: String };
const VERSION = "live-conversations-v1";

/** Enable the bridge only through the observer's explicit configuration. */
export function configureLiveCapture(options, cleanText) {
  state.options = { ...options };
  state.clean = cleanText;
}

/** Resolve a permitted run without reading model content from other agents. */
function findRun(event) {
  if (!state.options.liveExport || !event?.runId) return null;
  const existing = state.runs.get(event.runId);
  const key = event.sessionKey ?? existing?.sessionKey;
  if (typeof key !== "string" || !key.startsWith("agent:minilda:")) return null;
  if (state.options.sessionScope !== "all" && !key.startsWith("agent:minilda:phoenix-eval-")) return null;
  if (event.agentId && event.agentId !== "minilda") return null;
  const now = Date.now();
  // Bound the correlation cache; durable transcripts remain on disk.
  for (const [id, item] of state.runs) if (now - item.touchedAt > 3600000) state.runs.delete(id);
  const run = existing ?? {};
  run.sessionKey = key;
  run.sessionId = event.sessionId ?? run.sessionId;
  run.trigger = event.trigger ?? run.trigger;
  run.runId = event.runId;
  run.touchedAt = now;
  state.runs.set(event.runId, run);
  while (state.runs.size > 256) state.runs.delete(state.runs.keys().next().value);
  return run;
}

/** Save the current prompt separately from accumulated conversation history. */
export function captureLiveRecord(record) {
  const run = findRun(record?.context);
  if (run && record.hook === "llm_input" && run.prompt === undefined) run.prompt = record.event.prompt;
}

/** Produce an opaque conversation ID, keeping heartbeat and cron runs separate. */
export function buildSessionAttributes(event) {
  const run = findRun(event);
  if (!run) return {};
  const kind = run.trigger === "heartbeat" || run.sessionKey.endsWith(":heartbeat") ? "heartbeat"
    : run.sessionKey.includes(":cron:") ? "cron"
    : run.sessionKey.startsWith("agent:minilda:phoenix-eval-") ? "evaluation" : "conversation";
  const identity = run.sessionId ?? run.runId;
  const id = createHash("sha256").update(identity).digest("hex").slice(0, 24);
  return { "session.id": `openclaw:${kind}:${id}`, "openclaw.presentation.version": VERSION,
    "openclaw.presentation.kind": kind, "openclaw.presentation.session_source": run.sessionId ? "session_window" : "run_fallback" };
}

/** Read visible text blocks without tool calls or private signatures. */
function readVisibleText(message) {
  if (typeof message?.content === "string") return message.content;
  return Array.isArray(message?.content) ? message.content.filter((part) => part?.type === "text" && typeof part.text === "string").map((part) => part.text).join("\n\n") : "";
}

/** Read recorded reasoning from one provider response without retaining raw messages. */
function readReasoningTexts(message) {
  const texts = [message?.reasoning_content, message?.reasoning, message?.thinking];
  for (const part of Array.isArray(message?.content) ? message.content : []) {
    if (part && /^(thinking|reasoning)$/.test(part.type)) texts.push(part.thinking ?? part.reasoning ?? part.text);
  }
  return texts.filter((text) => typeof text === "string" && text.length);
}

/** Capture the exact provider-call output before native reasoning redaction. */
export function captureModelOutput(event, content, policy) {
  const run = findRun(event);
  if (!run) return { attrs: {} };
  const attrs = buildSessionAttributes(event);
  if (!policy.outputMessages) return { attrs };
  const messages = Array.isArray(content?.outputMessages) ? content.outputMessages : [];
  const assistant = messages.filter((message) => message?.role === "assistant").at(-1);
  run.finalText = assistant ? state.clean(readVisibleText(assistant)) : undefined;
  const texts = assistant ? readReasoningTexts(assistant) : [];
  const mode = state.options.captureReasoning;
  if (mode === "size" || mode === "text") {
    attrs["openclaw.presentation.reasoning_chars"] = texts.reduce((sum, text) => sum + text.length, 0);
    attrs["openclaw.presentation.reasoning_source"] = "native_provider_response";
  }
  return { attrs, reasoningText: mode === "text" && texts.length ? state.clean(texts.join("\n\n")) : undefined };
}

/** Preserve source tool names in normalized result messages without changing IDs or content. */
export function addToolResultNames(attributes, event, content, policy) {
  if (!findRun(event)) return;
  for (const direction of ["input", "output"]) {
    const field = `${direction}Messages`;
    const key = `gen_ai.${direction}.messages`;
    if (!policy[field] || typeof attributes[key] !== "string") continue;
    const names = new Map();
    for (const message of content?.[field] ?? []) {
      if (!["tool", "toolResult"].includes(message?.role)) continue;
      const id = message.toolCallId ?? message.tool_call_id;
      const name = message.toolName ?? message.name;
      if (typeof id !== "string" || typeof name !== "string" || !name) continue;
      names.set(id, names.has(id) && names.get(id) !== name ? null : name);
    }
    const messages = JSON.parse(attributes[key]);
    let changed = false;
    for (const message of messages) {
      if (message.role !== "tool" || message.name) continue;
      const results = message.parts?.filter((part) => part.type === "tool_call_response") ?? [];
      const name = results.length === 1 ? names.get(results[0].id) : undefined;
      if (!name) continue;
      message.name = state.clean(name);
      changed = true;
    }
    if (changed) {
      attributes[key] = JSON.stringify(messages);
      attributes[`${direction}.value`] = attributes[key];
    }
  }
}

/** Add a readable request and response to the existing interaction root. */
export function buildRunAttributes(event, policy) {
  const run = findRun(event);
  if (!run) return { attrs: {} };
  const attrs = { ...buildSessionAttributes(event), "openinference.span.kind": "AGENT",
    "openclaw.presentation.delivery": "unknown", "openclaw.presentation.output_source": "last_provider_response",
    "openclaw.presentation.original_name": "openclaw.harness.run" };
  if (policy.inputMessages) {
    attrs["input.value"] = run.prompt ?? "[Current prompt unavailable]";
    attrs["input.mime_type"] = "text/plain";
  }
  if (policy.outputMessages) {
    attrs["output.value"] = run.finalText || "[No final response captured]";
    attrs["output.mime_type"] = "text/plain";
  }
  const name = { heartbeat: "Heartbeat", conversation: "Conversation", cron: "Scheduled task", evaluation: "Evaluation" }[attrs["openclaw.presentation.kind"]];
  return { attrs, name };
}

/** Keep aggregate counters as diagnostics without charging model tokens twice. */
export function markAggregateUsage(attributes, event) {
  const prefix = state.options.sessionScope === "all" ? "agent:minilda:" : "agent:minilda:phoenix-eval-";
  if (!state.options.liveExport || !event?.sessionKey?.startsWith(prefix)) return;
  for (const key of Object.keys(attributes)) {
    if (key.startsWith("llm.token_count.") || key.startsWith("gen_ai.usage.") || key.startsWith("llm.cost.")) delete attributes[key];
  }
  attributes["openinference.span.kind"] = "CHAIN";
  attributes["openclaw.presentation.aggregate_usage"] = true;
}
