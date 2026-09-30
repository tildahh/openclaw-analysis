# OpenClaw conversation and evaluation observer

This passive plugin records OpenClaw assistant sessions. The internal agent ID
remains `minilda`. By default it records session keys starting with
`agent:minilda:phoenix-eval-` with a nonempty suffix. With `sessionScope: "all"`,
it also records ordinary conversations, heartbeats and scheduled tasks belonging
to that agent. It rejects any present
`agentId` other than `minilda`; the installed model-call hooks omit that field,
so their exact session-key prefix establishes agent identity. A run ID is also
required. Other agents are ignored before their content is processed.

This observer writes local JSONL files. The optional `liveExport` bridge also
enriches the installed native exporter through a small, version-checked patch.
It reuses native traces without creating duplicates or using the separate
historical `openclaw-conversations` project. Since September 29 at 11:51 UTC,
production conversations, heartbeats and scheduled messages go to `openclaw-live`;
isolated benchmark gateways continue using `openclaw-assistant`. Earlier traces
remain in their original projects.
Selected evaluation reconstructions use the `openclaw-assistant-evals` Phoenix
project. Native traces have content capture enabled
following the user's approval to upload all traces. Credential redaction is best
effort and does not anonymize arbitrary personal information in tool results.

Production routing uses only `diagnostics.otel.headers["x-project-name"] =
"openclaw-live"`. This exporter setting hot reloads at an idle boundary; it does
not require reinstalling this plugin or changing assistant behavior. See the
[routing audit](../runs/phoenix-live-routing-20260929T115004Z/change.json).

## Deployment configuration

Copy this directory to the Mac mini, then install/link it with the normal
OpenClaw local-plugin mechanism. Keep a timestamped backup of the installed
plugin, native exporter and configuration before deployment.
Merge this entry into existing configuration without replacing other plugins:

```json
{
  "plugins": {
    "entries": {
      "phoenix-eval-observer": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true,
          "allowPromptInjection": false
        },
        "config": {
          "outputDir": "/Users/minilda/.openclaw/phoenix-evals",
          "captureReasoning": "text",
          "sessionScope": "all",
          "liveExport": true
        }
      }
    }
  }
}
```

The manifest requests startup activation. `index.js` is the entry point. The
live bridge additionally requires `patch-native-exporter.py` to build the native
exporter patch for its exact recorded SHA-256. It refuses a different version.
See [live deployment and rollback](../docs/live-conversation-exporter.md).
The native exporter remains responsible for delivery to Phoenix and its content
capture policy is respected. Hot reload on this installed OpenClaw version
previously left a stale browser-plugin reference; perform a clean gateway restart
at an idle boundary when loading this pair of changes.

## Recorded evidence and limits

- `llm_input`: visible prompt/history, provider/model, image count, and a SHA-256
  hash plus character count of the system prompt. System prompt text, system and
  developer history, reasoning blocks, and image/audio/video payloads are omitted.
- `llm_output`: visible `assistantTexts`, provider/model, resolved reference,
  harness, and aggregate usage. Raw `lastAssistant` is omitted.
- Reasoning: omitted by default. With `"captureReasoning": "size"` in the plugin
  config the `llm_output` record gains `reasoningChars` (thinking blocks, `reasoning_content`
  and `reasoning`/`thinking` strings on the raw assistant message, counted, never stored);
  with `"text"` it also gains `reasoningTexts`, bounded and credential-redacted like every
  other string. `scripts/export_trace.py` always carries the count onto the AGENT root as
  `openclaw.reasoning_chars` and adds one CHAIN span per attempt with the text only when
  run with `--include-reasoning`. Reasoning is where personal detail is densest; enable
  `text` only when uploading recorded reasoning is intended. The live bridge
  adds a **Recorded reasoning** child for each provider response that contains
  reasoning, using the actual native response rather than duplicating this
  attempt-level hook onto every call. It is recorded text, not a complete account
  of the model's internal computation.
- `before_tool_call` / `after_tool_call`: tool name, call ID, sanitized params,
  visible result, error, and duration. Callbacks return no policy changes.
- `model_call_started` / `model_call_ended`: provider-call ID and metadata,
  timing and terminal outcome. These hooks have no raw model content.
- `agent_end`: success, error and duration; no full message array.

`llm_input` and `llm_output` describe an embedded **attempt**, not every provider
call. Do not duplicate their content onto individual model-call spans. Aggregate
usage belongs to the attempt. `agent_end.success` describes runtime completion,
not independently evaluated task correctness. Hook timestamps are observation
times; explicit durations are preferable when reconstructing durations.

Records contain diagnostic correlation IDs, not guaranteed native exporter span
IDs. Reconstructed OpenInference traces should have new IDs, the original run ID
as correlation metadata, `session.id`, explicit span kinds, and a clear
`post_run_reconstruction` label. Missing terminal events remain incomplete;
these best-effort hooks are not a durable delivery queue.

Tagged evaluations retain their existing JSONL location and schema. Ordinary
sessions go in the `live/` subdirectory, so existing batch collectors do not
mistake real conversations for evaluation runs. The live bridge keeps a bounded
in-memory map keyed by the exact run ID. It attaches the current prompt and last
provider response to the native interaction root, and hashes the actual session
window ID into Phoenix's `session.id`. Heartbeat and cron groups are separate
from conversations. A fallback run ID is explicitly labeled when the session ID
is unavailable. No text is inferred from timestamps or nearby interactions.

Files use hashed run IDs, directory mode `0700` and file mode `0600`. Content is
bounded to 16,000 characters per string, 100 items per array/object, and depth 8.
Long strings include an explicit truncation marker. These limits can truncate
evidence. The observer does not change a prompt,
tool input, model, memory, or result, and local capture failures fail open.

## Verification and sources

Run `npm test` in this directory. Tests use fixtures only, with no model calls,
OpenClaw startup, or exports.

Verified against installed OpenClaw `2026.9.5` (`ec9c1a1`):
[hook registration and permissions](https://github.com/openclaw/openclaw/blob/ec9c1a1/docs/plugins/hooks.md),
[attempt and provider-call boundaries](https://github.com/openclaw/openclaw/blob/ec9c1a1/docs/plugins/hooks/prompt-and-session.md),
[privacy and trace context](https://github.com/openclaw/openclaw/blob/ec9c1a1/docs/gateway/opentelemetry/privacy-and-trace-context.md).
Exact fields were also checked in installed `dist/runtime-api-BzC0x4-Q.d.ts`,
`dist/builtin-openclaw-C7-lQJ2Z.mjs`, and
`dist/attempt.model-diagnostic-events-cm5plRPy.mjs`.
