"""Convert explicitly reviewed OpenClaw hooks offline; send to Phoenix only with --export."""

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import urllib.error
import urllib.parse
import urllib.request

PRIVATE_KEYS = {"system", "systemprompt", "thinking", "reasoning", "signature", "apikey", "token", "accesstoken", "refreshtoken", "secret", "password", "authorization", "cookie", "credentials", "privatekey"}


def read_settings(path):
    """Read Phoenix endpoint and key as dotenv data, preserving environment overrides."""
    values = {key: os.environ[key] for key in ("PHOENIX_ENDPOINT", "PHOENIX_API_KEY") if os.environ.get(key)}
    if path.exists():
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if separator and key in {"PHOENIX_ENDPOINT", "PHOENIX_API_KEY"} and key not in values:
                parts = shlex.split(value, comments=True)
                if parts:
                    values[key] = parts[0]
    return values


def sanitize_value(value):
    """Remove known private fields and credential patterns from already reviewed content."""
    if isinstance(value, dict):
        if value.get("type") in {"thinking", "reasoning", "redacted_thinking", "image", "image_url", "input_image", "audio", "input_audio", "video", "file"} or value.get("role") in {"system", "developer"}:
            return None
        return {key: sanitize_value(item) for key, item in value.items() if not any(re.sub(r"[^a-z]", "", key.lower()).endswith(secret) for secret in PRIVATE_KEYS)}
    if isinstance(value, list):
        return [clean for item in value if (clean := sanitize_value(item)) is not None]
    if isinstance(value, str):
        if re.search(r"-----BEGIN [^-]*PRIVATE KEY-----", value):
            return "[REDACTED PRIVATE KEY]"
        value = re.sub(r"(?is)<(?:think|thinking|analysis)>.*?</(?:think|thinking|analysis)>", "", value)
        value = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)(\b[\w-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization|cookie)\b\s*[:=]\s*)([^\r\n,;]+)", r"\1[REDACTED]", value)
        return re.sub(r"\b(?:sk-|phx_)[A-Za-z0-9_-]{12,}\b", "[REDACTED KEY]", value)
    return value


def read_visible_text(value):
    """Extract only visible text blocks, excluding system roles and reasoning blocks."""
    value = sanitize_value(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := read_visible_text(item)))
    if isinstance(value, dict):
        if value.get("type") in {"text", "input_text", "output_text"}:
            return read_visible_text(value.get("text", ""))
        if value.get("role") in {"user", "assistant", "tool", "toolResult"}:
            return read_visible_text(value.get("content", ""))
    return ""


def format_history(messages):
    """Preserve speaker labels on reviewed visible history so claims keep their attribution."""
    return [f"[{message.get('role', 'history') if isinstance(message, dict) else 'history'}] {text}"
            for message in messages if (text := read_visible_text(message))]


def build_attributes(values):
    """Encode a small attribute mapping using OTLP JSON scalar values."""
    return [{"key": key, "value": {"stringValue": str(value)}} for key, value in values.items() if value is not None]


def parse_timestamp(value):
    """Convert an explicit timezone-bearing ISO timestamp to epoch nanoseconds."""
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Hook timestamps must include a timezone")
    return int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1000


def build_payload(records, case, variant, project, source_sha, include_reasoning=False):
    """Reconstruct one session's root and observed model/tool calls without inventing call content.

    Reasoning recorded by the observer (captureReasoning: size or text) is summarized as a character count on the root;
    the text itself becomes a CHAIN child span per attempt only when include_reasoning is set.
    """
    if not records:
        raise ValueError("Reviewed events file is empty")
    records = sorted(records, key=lambda record: parse_timestamp(record["at"]))
    sessions = {record.get("context", {}).get("sessionId") or record["event"].get("sessionId") for record in records} - {None}
    if len(sessions) != 1:
        raise ValueError("Reviewed events must identify exactly one sessionId")
    session = sessions.pop()
    trace_id, root_id = secrets.token_hex(16), secrets.token_hex(8)
    start, end = parse_timestamp(records[0]["at"]), parse_timestamp(records[-1]["at"])
    run_ids = sorted({record.get("context", {}).get("runId") or record["event"].get("runId") for record in records} - {None})
    metadata = {"source": "reconstructed_from_runtime_hooks", "case": case, "variant": variant, "run_ids": run_ids}
    common = {"session.id": session, "metadata": json.dumps(metadata)}
    root = {"traceId": trace_id, "spanId": root_id, "name": f"OpenClaw · {case} · {variant}", "kind": 1}
    spans, pending, prompts, answers, history, models, usage = [root], defaultdict(deque), [], [], [], set(), Counter()
    incomplete, errors, attempts = 0, 0, 0
    reasoning_chars, reasoning_spans, attempt_started = None, 0, None
    for record in records:
        hook, event, at = record["hook"], record["event"], parse_timestamp(record["at"])
        if hook == "llm_input":
            attempts += 1
            attempt_started = at
            if attempts == 1:
                history = format_history(event.get("historyMessages", []))
            prompts.append(read_visible_text(event.get("prompt", "")))
        elif hook == "llm_output":
            answers.append(read_visible_text(event.get("assistantTexts", [])))
            if isinstance(event.get("reasoningChars"), (int, float)) and not isinstance(event.get("reasoningChars"), bool):
                reasoning_chars = (reasoning_chars or 0) + int(event["reasoningChars"])
            texts = [text for text in event.get("reasoningTexts", []) if isinstance(text, str) and text]
            if include_reasoning and texts:
                reasoning_spans += 1
                began = attempt_started if attempt_started is not None and attempt_started <= at else at
                spans.append({"traceId": trace_id, "spanId": secrets.token_hex(8), "parentSpanId": root_id, "name": "Reasoning · attempt " + str(attempts), "kind": 1,
                              "startTimeUnixNano": str(began), "endTimeUnixNano": str(at), "status": {"code": 1},
                              "attributes": build_attributes({**common, "openinference.span.kind": "CHAIN", "openclaw.timing_source": "attempt_window",
                                                              "openclaw.reasoning_chars": sum(len(text) for text in texts),
                                                              "input.value": "", "input.mime_type": "text/plain",
                                                              "output.value": "\n\n".join(texts), "output.mime_type": "text/plain"})})
            usage.update({key: value for key, value in event.get("usage", {}).items() if key in {"input", "output", "cacheRead", "cacheWrite", "total"} and isinstance(value, (int, float))})
        if event.get("model"):
            models.add(event["model"])
        if hook in {"before_tool_call", "model_call_started"}:
            kind = "TOOL" if hook == "before_tool_call" else "LLM"
            key = (kind, event.get("toolCallId") if kind == "TOOL" else event.get("callId"))
            if not key[1]:
                key = (kind, event.get("runId"), event.get("toolName"))
            pending[key].append(record)
        elif hook in {"after_tool_call", "model_call_ended"}:
            kind = "TOOL" if hook == "after_tool_call" else "LLM"
            key = (kind, event.get("toolCallId") if kind == "TOOL" else event.get("callId"))
            if not key[1]:
                key = (kind, event.get("runId"), event.get("toolName"))
            before = pending[key].popleft() if pending[key] else None
            duration = event.get("durationMs")
            reported_duration = isinstance(duration, (int, float)) and not isinstance(duration, bool)
            began = at - int(duration * 1_000_000) if reported_duration else parse_timestamp(before["at"]) if before else at
            if began > at or began < 0:
                raise ValueError("Invalid hook duration")
            incomplete += int(before is None)
            start = min(start, began)
            tool_result = event.get("result")
            tool_failed = isinstance(tool_result, dict) and (
                bool(tool_result.get("isError")) or tool_result.get("status") == "error"
                or (isinstance(tool_result.get("details"), dict) and tool_result["details"].get("status") == "error")
            )
            failed = bool(event.get("error")) or event.get("outcome") == "error" or (kind == "TOOL" and tool_failed)
            errors += int(failed)
            run_id = record.get("context", {}).get("runId") or event.get("runId")
            attributes = {**common, "metadata": json.dumps({**metadata, "run_id": run_id}), "openinference.span.kind": kind, "openclaw.timing_source": "reported_duration" if reported_duration else "paired_hook_window" if before else "end_hook_only"}
            if kind == "TOOL":
                params = (before or record)["event"].get("params", {})
                result = {"result": event.get("result"), "error": event["error"]} if event.get("error") else event.get("result")
                attributes.update({"tool.name": event.get("toolName"), "input.value": json.dumps(sanitize_value(params)), "input.mime_type": "application/json", "output.value": json.dumps(sanitize_value(result)), "output.mime_type": "application/json"})
                name = "Tool · " + str(event.get("toolName", "unknown"))
            else:
                attributes.update({"llm.model_name": event.get("model"), "llm.provider": event.get("provider"), "openclaw.call_id": event.get("callId"), "openclaw.error_category": event.get("errorCategory")})
                name = "Model · " + str(event.get("model", "unknown"))
            spans.append({"traceId": trace_id, "spanId": secrets.token_hex(8), "parentSpanId": root_id, "name": name, "kind": 1, "startTimeUnixNano": str(began), "endTimeUnixNano": str(at), "attributes": build_attributes(attributes), "status": {"code": 2 if failed else 1}})
    unclosed = sum(len(items) for items in pending.values())
    ends = [record["event"] for record in records if record["hook"] == "agent_end"]
    root_failed = any(event.get("success") is False or event.get("error") for event in ends)
    requests = [("[latest request]" if index == len(prompts) - 1 else "[earlier request]") + "\n" + prompt for index, prompt in enumerate(prompts) if prompt]
    root.update(startTimeUnixNano=str(start), endTimeUnixNano=str(end), status={"code": 2 if root_failed else 1 if ends else 0}, attributes=build_attributes({**common, "openinference.span.kind": "AGENT", "openclaw.reasoning_chars": reasoning_chars, "input.value": "\n\n".join(history + requests), "input.mime_type": "text/plain", "output.value": answers[-1] if answers else "", "output.mime_type": "text/plain"}))
    resource = build_attributes({"service.name": "openclaw-reviewed-hooks", "openinference.project.name": project})
    payload = {"resourceSpans": [{"resource": {"attributes": resource}, "scopeSpans": [{"scope": {"name": "openclaw-reviewed-hook-converter"}, "spans": spans}]}]}
    kinds = Counter(next(item["value"]["stringValue"] for item in span["attributes"] if item["key"] == "openinference.span.kind") for span in spans)
    summary = {**metadata, "project": project, "source_sha256": source_sha, "trace_id": trace_id, "session_id": session, "span_count": len(spans), "kinds": dict(kinds), "duration_ms": (end - start) / 1_000_000, "tool_count": kinds["TOOL"], "model_call_count": kinds["LLM"], "model_names": sorted(models), "attempt_count": attempts, "reported_attempt_usage": dict(usage), "call_error_count": errors, "root_failed": root_failed, "unpaired_end_hooks": incomplete, "unclosed_start_hooks": unclosed, "reasoning_chars": reasoning_chars, "reasoning_spans": reasoning_spans, "include_reasoning": include_reasoning, "trace_link": "/redirects/traces/" + trace_id, "exported": False}
    return payload, summary


def save_payload(events_path, output, case, variant, project, include_reasoning=False):
    """Persist a payload before export and reuse its identifiers on an identical retry."""
    raw = events_path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    summary_path = output.with_suffix(".summary.json")
    if output.exists():
        payload, summary = json.loads(output.read_text()), json.loads(summary_path.read_text())
        if any(summary.get(key) != value for key, value in {"source_sha256": fingerprint, "case": case, "variant": variant, "project": project, "include_reasoning": include_reasoning}.items()):
            raise ValueError("Existing payload belongs to different inputs; choose a new --output path")
    else:
        records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        payload, summary = build_payload(records, case, variant, project, fingerprint, include_reasoning)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return payload, summary, summary_path


def copy_attributes(source, target):
    """Copy saved scalar attributes into protobuf without interpreting identifier bytes."""
    fields = {"stringValue": ("string_value", str), "intValue": ("int_value", int),
              "boolValue": ("bool_value", bool), "doubleValue": ("double_value", float)}
    for item in source:
        kind, value = next(iter(item["value"].items()))
        field, convert = fields[kind]
        attribute = target.add(key=item["key"])
        setattr(attribute.value, field, convert(value))


def serialize_payload(payload):
    """Encode the saved OTLP fixture as protobuf, converting hex IDs directly to bytes."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    request = ExportTraceServiceRequest()
    for resource in payload["resourceSpans"]:
        target_resource = request.resource_spans.add()
        copy_attributes(resource["resource"]["attributes"], target_resource.resource.attributes)
        for scope in resource["scopeSpans"]:
            target_scope = target_resource.scope_spans.add()
            target_scope.scope.name = scope.get("scope", {}).get("name", "")
            for saved in scope["spans"]:
                trace_id, span_id = bytes.fromhex(saved["traceId"]), bytes.fromhex(saved["spanId"])
                parent_id = bytes.fromhex(saved.get("parentSpanId", ""))
                if len(trace_id) != 16 or len(span_id) != 8 or len(parent_id) not in {0, 8}:
                    raise ValueError("Invalid saved trace or span identifier length")
                span = target_scope.spans.add(trace_id=trace_id, span_id=span_id, parent_span_id=parent_id,
                    name=saved["name"], kind=int(saved["kind"]),
                    start_time_unix_nano=int(saved["startTimeUnixNano"]), end_time_unix_nano=int(saved["endTimeUnixNano"]))
                copy_attributes(saved["attributes"], span.attributes)
                span.status.code = int(saved.get("status", {}).get("code", 0))
                span.status.message = saved.get("status", {}).get("message", "")
    return request.SerializeToString()


def export_payload(payload, settings, project):
    """Send the saved payload as OTLP protobuf and validate the binary collector response."""
    from google.protobuf.message import DecodeError
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

    base = settings["PHOENIX_ENDPOINT"].rstrip("/").removesuffix("/v1/traces")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("Invalid Phoenix endpoint")
    headers = {"Content-Type": "application/x-protobuf", "Accept": "application/x-protobuf", "x-project-name": project}
    if settings.get("PHOENIX_API_KEY"):
        headers["Authorization"] = "Bearer " + settings["PHOENIX_API_KEY"]
    request = urllib.request.Request(base + "/v1/traces", data=serialize_payload(payload), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        try:
            result = ExportTraceServiceResponse.FromString(response.read())
        except DecodeError:
            raise ValueError("Invalid Phoenix protobuf response") from None
    if result.partial_success.rejected_spans:
        raise ValueError("Phoenix rejected spans")
    return base


def run_converter():
    """Convert only a caller-reviewed file and require an explicit flag for network export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-events", type=Path, required=True, help="Caller-reviewed, sanitized JSONL for exactly one selected test session")
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--project", default="openclaw-assistant-evals")
    parser.add_argument("--output", type=Path, help="Persisted OTLP payload; identical retries reuse its trace/span IDs")
    parser.add_argument("--export", action="store_true", help="Explicitly send the saved reviewed payload to Phoenix")
    parser.add_argument("--include-reasoning", action="store_true", help="Add a CHAIN span per attempt carrying reasoning text the observer captured (captureReasoning: text); the count is always kept")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    args = parser.parse_args()
    try:
        payload, summary, summary_path = save_payload(args.reviewed_events, args.output or args.reviewed_events.with_suffix(".otlp.json"), args.case, args.variant, args.project, args.include_reasoning)
        if args.export:
            base = export_payload(payload, read_settings(args.env_file), args.project)
            summary.update(exported=True, trace_link=base + "/redirects/traces/" + summary["trace_id"])
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except urllib.error.HTTPError as error:
        parser.exit(1, f"Phoenix export failed: HTTP {error.code}; saved payload can be retried.\n")
    except ImportError:
        parser.exit(1, "Protobuf export requires opentelemetry-proto and protobuf; use ../cadbench/.venv/bin/python.\n")
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Conversion/export failed; check reviewed schema, output pairing, and endpoint settings. No event or credential values displayed.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_converter()
