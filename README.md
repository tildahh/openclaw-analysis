# Evaluating an always-on OpenClaw assistant with Phoenix

Start with the cookbook: [COOKBOOK.md](COOKBOOK.md). It traces an OpenClaw assistant in Arize Phoenix, builds a labeled heartbeat dataset, checks an LLM judge against human labels, and tests harness and model changes as Phoenix experiments.

## How the traces work in Phoenix

- **Live traces:** the [observer plugin](observer/) and its [exporter patch](observer/patch-native-exporter.py) add the question, the answer, a session ID and readable root names to the spans OpenClaw already sends, so each run reads clearly in Phoenix. Setup and rollback: [docs/live-conversation-exporter.md](docs/live-conversation-exporter.md). A walkthrough of the code changes: [docs/exporter-deep-dive.ipynb](docs/exporter-deep-dive.ipynb).
- **Earlier conversations:** before the live patch, [scripts/export_trace.py](scripts/export_trace.py) rebuilt readable Phoenix traces offline from recorded OpenClaw hook events. Guide: [docs/conversation-tracing.md](docs/conversation-tracing.md).
