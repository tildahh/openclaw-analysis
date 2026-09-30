# Readable conversations in Phoenix

This page documents the earlier **historical reconstruction** workflow. New live
interactions now use the [live conversation integration](live-conversation-exporter.md)
in `openclaw-assistant`. The statements below about leaving the installed exporter
unchanged describe the original reconstruction work, before that later deployment.

The conversation view puts the current question and final answer at the top of a trace. Expand its children to inspect model calls, tool arguments and results, and recorded reasoning. Use **View Session → Turns** to read the conversation like a chat; use **View trace** to inspect a turn.

Verified examples in the separate **openclaw-conversations** project:

| Example | What is included |
| --- | --- |
| [This week's coursework question](https://app.phoenix.arize.com/s/matildaorona/redirects/traces/da9569e8e8e038c0bc4ac6048f5e0dc8) | Original question and answer, 5 model calls, 9 tool calls, 5 recorded-reasoning steps, and a completed delivery record linked by the original trace ID. |
| [Coursework conversation](https://app.phoenix.arize.com/s/matildaorona/projects/UHJvamVjdDozMQ==/sessions/UHJvamVjdFNlc3Npb246MzA=?timeRangeKey=7d) | Phoenix's chat-style Turns view, currently containing the selected exported question and answer. |
| [Later scheduled coursework reminder](https://app.phoenix.arize.com/s/matildaorona/redirects/traces/d9a7e494369fd1ed40737c484de0c4c6) | The heartbeat prompt, final reminder, 7 model calls, 11 tool calls, and 7 recorded-reasoning steps. Telegram delivery is unknown. This is a separate heartbeat session. |

These are readable copies of saved interactions, not new assistant runs or additional experimental results. Each root has a **Saved interaction** annotation explaining its origin and linking to the original trace.

## What changed from the default export

The installed OpenClaw exporter remains unchanged. Two local scripts provide an export adapter: `scripts/collect_conversation.py` reads an existing native trace and its matching saved transcript; `scripts/export_conversation.py` creates the readable view. This preserves the current observation experiment and requires no gateway restart.

| Default native export we inspected | What the adapter adds or changes |
| --- | --- |
| Several operational wrapper spans; root has no visible request or answer | One AGENT root displaying the current question and final answer |
| No conversation identifier for Phoenix Sessions | `session.id` based on the actual OpenClaw session window and trigger |
| Technical model and tool span names | Ordered model and tool steps under the interaction root; original call content and timing retained |
| Reasoning absent from the exported calls | Optional **Recorded reasoning** children, matched to saved transcript responses |
| Aggregate usage wrapper alongside individual model calls | Only individual calls carry token totals; the aggregate wrapper is not copied |
| Delivery can appear separately from the response | Delivery included only when it has the same native trace ID; otherwise explicitly unknown |
| Original trace and span identifiers | New identifiers for the readable copy, with original identifiers retained for comparison |

One trace represents one interaction. Multiple exported turns from the same actual chat session use the same `session.id`. A reset starts a different session. Heartbeats use a different group from direct messages, even if older runs shared an underlying session; each newly isolated heartbeat has its own session UUID. This affects display only, not what the assistant remembers.

This is currently a selected-interaction export, **not an automatic live replacement**. Future completed interactions can use the same collector and exporter. Only the two examples above have been exported with this adapter. Multi-turn identifier consistency is tested, but the live session example currently contains one selected turn.

## How another saved interaction is exported

From the repository directory, using the existing Python environment:

```bash
../cadbench/.venv/bin/python -m scripts.collect_conversation \
  --trace-id NATIVE_TRACE_ID --output-dir runs/conversation-EXAMPLE

../cadbench/.venv/bin/python -m scripts.export_conversation \
  --native-spans runs/conversation-EXAMPLE/native.json \
  --evidence runs/conversation-EXAMPLE/evidence.json \
  --title 'A short description of the question' \
  --output runs/conversation-EXAMPLE/view.otlp.json \
  --include-reasoning
```

The first command reads Phoenix and the mini's SQLite transcript database in read-only mode. The second builds a local payload. Inspect it, then repeat the second command with `--export` to publish. Neither command invokes the assistant. Raw evidence and payload files are private local files under ignored `runs/` directories. Existing credential masking remains in place; review is still needed before publishing private material.

The collector requires an unambiguous match: actual run ID, exact tool-call IDs and names, visible response text, and matching model-start times. It also verifies that the selected current question appears in the first model input. Missing or conflicting evidence stops the export rather than attaching a guessed conversation. It currently requires a complete captured run and the saved transcript schema observed on this OpenClaw build.

Identical inputs and export options reproduce the same trace and span identifiers. Keep the title, project, evidence, and reasoning option unchanged when retrying. Changing them creates a different readable copy; do not count such copies as independent runs. Existing output files cannot be silently replaced with a different payload.

## Limits that matter for analysis

- **Recorded reasoning is model-generated text saved by OpenClaw**, not a complete record of the model's internal computation. Each reasoning child's time window is its parent call's window, not extra work to add to runtime. No tokens are added for these children.
- Full system prompts and unavailable original data are not recovered. Existing content truncation is preserved, not repaired by guessing.
- A completed run means it finished executing. It does not mean its answer is correct. A delivery completion does not prove the user read the message.
- The separate project prevents these readable copies from inflating the original project's metrics. Do not combine both projects when counting attempts.
- The user explicitly said the coursework reminder was welcome and appreciated the absence of `NO_REPLY` prose. It is labeled **Useful reminder**. Repetition alone is not evidence of an unwanted interruption. Earlier judge results were not silently rewritten.

## Verification and audit

The original coursework question and answer, an expandable reasoning step, a calendar tool's arguments and returned error, and the chat-style Session view were verified in the Phoenix UI. Both exported traces were read back through the API: 21 spans for the direct interaction and 26 for the heartbeat, with separate session identifiers.

The exporter test suite passed **15 tests**, including exact matching, rejection of mismatched records, session grouping, heartbeat separation, reasoning opt-in, deterministic identifiers, token accounting, and a read-only SQLite collector fixture. Tests make no model calls.

```bash
../cadbench/.venv/bin/python -m unittest discover -s tests -p 'test_export*.py'
```

The audit is in `runs/conversation-tracing-20260929T0035Z/change.json`, alongside native source snapshots, exported payloads, and API readbacks. The installed diagnostics exporter is version **2026.9.5**. Its inspected runtime file has SHA-256 `f0366f582292f8744bab8ae4e428f2e92d00e542f9fd9753a573105fe4e5cf21`; a copy is preserved as `native-exporter-original.mjs`. No live exporter files, model settings, heartbeat settings, assistant instructions, or schedules were changed for this work.

To stop using this view, use the original project. There is no assistant configuration to roll back. Keep the derived project separate and preserve source evidence.

The cookbook lesson is concrete: first make the user's request, the assistant's actions, and the result visible together. Conversation grouping helps people navigate; exact source matching keeps that convenience from becoming misleading evidence.
