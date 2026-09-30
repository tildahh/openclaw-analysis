import { createHash } from "node:crypto";
import { appendFileSync, chmodSync, closeSync, constants, fchmodSync, mkdirSync, openSync } from "node:fs";
import { join } from "node:path";
import { captureLiveRecord, configureLiveCapture } from "./live-trace.js";

const PREFIX = "agent:minilda:phoenix-eval-";
const DEFAULT_DIR = "/Users/minilda/.openclaw/phoenix-evals";
const OMIT = /^(systemPrompt|system_prompt|thinking|reasoning|redacted_thinking|reasoning_content|thoughtSignature|signature|image|images|image_url|audio|video|base64|bytes)$/i;
const SECRET = /(?:authorization|apikey|accesstoken|refreshtoken|password|passwd|secret|secretaccesskey|privatekey|token|credentials?|cookies?)$/i;
const FIELDS = {
  llm_input: ["provider", "model", "prompt", "historyMessages", "imagesCount"],
  llm_output: ["provider", "model", "resolvedRef", "harnessId", "assistantTexts", "usage"],
  before_tool_call: ["toolName", "toolCallId", "params"],
  after_tool_call: ["toolName", "toolCallId", "params", "result", "error", "durationMs"],
  model_call_started: ["callId", "provider", "model", "api", "transport", "contextTokenBudget"],
  model_call_ended: ["callId", "provider", "model", "api", "transport", "durationMs", "outcome", "errorCategory", "failureKind", "requestPayloadBytes", "responseStreamBytes", "timeToFirstByteMs", "upstreamRequestIdHash"],
  agent_end: ["success", "error", "durationMs"],
};

/** Resolve tagged evaluations or explicitly enabled ordinary Minilda sessions. */
export function resolveScope(event = {}, context = {}, options = {}) {
  const sessionKey = context.sessionKey ?? event.sessionKey;
  const prefix = options.sessionScope === "all" ? "agent:minilda:" : PREFIX;
  if (typeof sessionKey !== "string" || !sessionKey.startsWith(prefix) || sessionKey.length === prefix.length) return null;
  if (context.agentId !== undefined && context.agentId !== "minilda") return null;
  const runId = context.runId ?? event.runId;
  if (typeof runId !== "string" || !runId) return null;
  return { agentId: "minilda", sessionKey, sessionId: context.sessionId ?? event.sessionId, runId, trace: context.trace };
}

/** Redact common credential literals and bound a visible text field. */
export function redactText(value) {
  const redacted = value
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(/\b(?:sk|rk)-[A-Za-z0-9_-]{12,}/g, "[REDACTED]")
    .replace(/\b((?:[A-Z0-9_]{0,128}(?:API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|PASSWORD|SECRET)|api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,;]+)/gi, "$1[REDACTED]")
    .replace(/https?:\/\/[^\s/@:]+:[^\s/@]+@/gi, "https://[REDACTED]@")
    .replace(/data:(?:image|audio|video)\/[^\s"']+/gi, "[BINARY OMITTED]");
  return redacted.length > 16000 ? redacted.slice(0, 16000) + `\n[TRUNCATED: ${redacted.length - 16000} additional characters omitted]` : redacted;
}

/** Retain bounded visible content while omitting private reasoning and binary payloads. */
export function sanitizeValue(value, depth = 0) {
  if (depth > 8) return "[DEPTH LIMIT]";
  if (typeof value === "string") return redactText(value);
  if (value === null || typeof value === "number" || typeof value === "boolean") return value;
  if (!value || typeof value !== "object" || ArrayBuffer.isView(value)) return undefined;
  if (Array.isArray(value)) return value.slice(0, 100).map((item) => sanitizeValue(item, depth + 1)).filter((item) => item !== undefined);
  if (["system", "developer"].includes(value.role)) return undefined;
  if (typeof value.type === "string" && /thinking|reasoning|image|audio|video|redacted_thinking/i.test(value.type)) return undefined;
  return Object.fromEntries(Object.entries(value).slice(0, 100).flatMap(([key, item]) => {
    if (OMIT.test(key)) return [];
    const clean = SECRET.test(key.replace(/[-_]/g, "")) ? "[REDACTED]" : sanitizeValue(item, depth + 1);
    return clean === undefined ? [] : [[key, clean]];
  }));
}

/** Collect reasoning text from the raw assistant message in the shapes the runtime and OpenAI-compatible servers use. */
export function extractReasoning(message) {
  const texts = [];
  const push = (value) => { if (typeof value === "string" && value.length) texts.push(value); };
  if (message && typeof message === "object") {
    push(message.reasoning_content);
    push(message.reasoning);
    push(message.thinking);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (block && typeof block === "object" && /thinking|reasoning/i.test(String(block.type ?? ""))) push(block.thinking ?? block.reasoning ?? block.text ?? block.content);
      }
    }
  }
  return { chars: texts.reduce((total, text) => total + text.length, 0), texts };
}

/** Build a scoped record without mutating any host event or message. `options.captureReasoning` is "none" (default), "size" or "text". */
export function buildRecord(hook, event, context, at = new Date().toISOString(), options = {}) {
  const scope = resolveScope(event, context, options);
  if (!scope || !FIELDS[hook]) return null;
  const selected = Object.fromEntries(FIELDS[hook].filter((key) => event[key] !== undefined).map((key) => [key, event[key]]));
  if (hook === "llm_input" && typeof event.systemPrompt === "string") {
    selected.systemPromptSha256 = createHash("sha256").update(event.systemPrompt).digest("hex");
    selected.systemPromptChars = event.systemPrompt.length;
  }
  const mode = options.captureReasoning === "text" ? "text" : options.captureReasoning === "size" ? "size" : "none";
  if (hook === "llm_output" && mode !== "none") {
    const reasoning = extractReasoning(event.lastAssistant);
    selected.reasoningChars = reasoning.chars;
    if (mode === "text") selected.reasoningTexts = reasoning.texts;
  }
  return { hook, at, context: sanitizeValue({ ...scope, ...(context.trigger ? { trigger: context.trigger } : {}) }), event: sanitizeValue(selected) };
}

/** Append one private JSONL record, refusing a symlink as its destination. */
export function writeRecord(directory, record) {
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  chmodSync(directory, 0o700);
  const filename = createHash("sha256").update(record.context.runId).digest("hex").slice(0, 24) + ".jsonl";
  const path = join(directory, filename);
  const fd = openSync(path, constants.O_WRONLY | constants.O_CREAT | constants.O_APPEND | constants.O_NOFOLLOW, 0o600);
  try {
    fchmodSync(fd, 0o600);
    appendFileSync(fd, JSON.stringify(record) + "\n");
  } finally { closeSync(fd); }
}

/** Create a passive callback whose own I/O errors cannot block a tool or agent turn. */
export function createObserver(hook, directory, logger, writer = writeRecord, options = {}) {
  /** Record a selected event and fail open if local capture fails. */
  return function observeEvent(event, context) {
    try {
      const record = buildRecord(hook, event, context, undefined, options);
      if (record) {
        captureLiveRecord(record);
        // Preserve the eval directory's existing meaning for batch collectors.
        const destination = options.sessionScope === "all" && !record.context.sessionKey.startsWith(PREFIX) ? join(directory, "live") : directory;
        writer(destination, record);
      }
    } catch {
      // Deliberately omit exception details: they can contain paths or event text.
      try { logger?.warn?.(`phoenix-eval-observer: local capture failed for ${hook}`); } catch {}
    }
  };
}

/** Register local observation hooks without installing tools or modifying prompts. */
function registerObserver(api) {
  const directory = api.pluginConfig?.outputDir || DEFAULT_DIR;
  const options = { captureReasoning: api.pluginConfig?.captureReasoning ?? "none", sessionScope: api.pluginConfig?.sessionScope ?? "eval", liveExport: api.pluginConfig?.liveExport === true };
  configureLiveCapture(options, redactText);
  for (const hook of Object.keys(FIELDS)) api.on(hook, createObserver(hook, directory, api.logger, undefined, options));
}

export default {
  id: "phoenix-eval-observer",
  name: "Phoenix Conversation Observer",
  register: registerObserver,
};
