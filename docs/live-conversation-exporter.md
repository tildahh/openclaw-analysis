# Live conversation traces

The source lives in this repository's `observer/` directory. It runs on the Mac
mini alongside OpenClaw. A full OpenClaw source checkout is not needed for this
experiment. We keep the exact upstream exporter hash, a small patch builder,
tests, and a timestamped deployment backup instead.

## What the change does

One interaction keeps its original native trace ID. Its `openclaw.harness.run`
root becomes an AGENT span named **Conversation**, **Heartbeat**, **Scheduled task**
or **Evaluation**, with the current request and the last model response
visible in Phoenix's Input and Output panels. The model and tool calls retain
their original IDs, parents, timings and content. A **Recorded reasoning** child
appears under a model call when that provider response includes reasoning text.
It is a zero-duration observation marker and adds no tokens or runtime.

Ordinary turns from the same actual OpenClaw session window share an opaque
`session.id`. Phoenix Sessions can therefore show the turns together. Isolated
heartbeats have separate session windows and separate groups. Scheduled cron
work is also labeled separately. This changes the display, not the history that
the assistant reads.

```mermaid
flowchart TD
    Request[User message or scheduled heartbeat] --> Run[Original OpenClaw interaction]
    Run --> Hooks[Observer: current request and session identity]
    Run --> Native[Native exporter: model and tool calls]
    Hooks --> View[Small presentation bridge]
    Native --> View
    View --> Phoenix[Configured Phoenix project]
    Phoenix --> Root[Request and response at trace root]
    Phoenix --> Steps[Model calls, recorded reasoning, and tools]
    Phoenix --> Sessions[Conversation turns grouped in Sessions]
```

## Exact differences from the default exporter

| Area | Change |
| --- | --- |
| Observer scope | Explicit `sessionScope: "all"` includes this assistant's conversations, heartbeats, cron work and tagged evaluations. Other agents remain excluded. |
| Root Input | Current attempt prompt from `llm_input`; conversation history is not substituted for the current request. Missing input is identified explicitly. |
| Root Output | Last provider response's visible text; missing output is identified explicitly. This is not proof of Telegram delivery. |
| Span kind and name | Root labeled AGENT and given a readable interaction name, with its original name saved in an attribute. Native model/tool span names stay intact for existing analysis filters. |
| Conversation grouping | Hash of the actual session window ID in `session.id`, with separate conversation, heartbeat, cron and evaluation prefixes. No raw routing identifiers added. |
| Recorded reasoning | Opt-in text from each native provider response before default exporter redaction, with credential masking and a 16,000-character limit. Native content policy must allow output capture. No private signatures are exported. |
| Tool-result labels | Preserve the source `toolName` as the normalized result message's `name`, joined by exact call ID. Phoenix shows `tool: memory_search`, for example. The inner `Tool Result: <ID>` row remains Phoenix's standard ID display. Tool IDs, result bodies and native span names are unchanged. |
| Token accounting | Individual model calls retain their usage. Aggregate usage remains as diagnostic counters but stops being counted a second time as an LLM call. |
| Local records | Existing evaluation JSONL files unchanged; other sessions saved under `phoenix-evals/live/`. |

The patch does not change prompts, tools, model configuration, memory, heartbeat
isolation, schedules, routing, or account actions. It does not delete or rewrite
old Phoenix traces or notes. The earlier `openclaw-conversations` examples remain
historical reconstructions. You can route everyday activity and evaluation runs to separate Phoenix projects with the
`diagnostics.otel.headers["x-project-name"]` setting. Keep those populations separate
when comparing results.

## Installation and maintenance

The supported installed version is OpenClaw **2026.9.5 (`ec9c1a1`)**, with
`@openclaw/diagnostics-otel` 2026.9.5. The upstream runtime file SHA-256 is
`f0366f582292f8744bab8ae4e428f2e92d00e542f9fd9753a573105fe4e5cf21`.

1. Back up the installed observer, native runtime file and configuration.
2. Run `npm test` in `observer/` locally and in a staging directory on the mini.
3. Build a patched copy with `observer/patch-native-exporter.py`, supplying the
   exact upstream source file, output path and installed observer directory.
4. At an idle boundary, install the observer files and patched native runtime.
   Set only `sessionScope: "all"` and `liveExport: true`; preserve other settings.
5. Restart the gateway, check plugin and gateway health, then verify two ordinary
   non-delivering turns share one Phoenix session but have separate trace IDs.
6. Read back the spans, inspect the question, response, tool and reasoning panels,
   and record the outcome in the deployment audit.

The native patch is local, not an upstream OpenClaw feature. An OpenClaw plugin
update may replace it. Do not apply it blindly to a new version: the hash guard
requires a source review and rebuilt patch. Keep the observer and native patch
installed together.

Rollback: at an idle boundary, restore the backed-up native runtime and observer
directory; restore only the observer's two changed config keys (or its prior
entry if nothing else changed); restart and verify health. Avoid restoring the
whole config file over unrelated changes made after deployment.

## Limits for the cookbook

An exported response is not a sent message, and completed execution is not a
correct answer. Delivery must be verified independently. Recorded reasoning is
model-produced text, not complete internal computation. Existing content limits
and redaction still apply. This improves new live traces only; historical traces
need the separately labeled reconstruction workflow. Compare experiment results
across this deployment using explicit timestamps because the reporting format
and token accounting changed.
