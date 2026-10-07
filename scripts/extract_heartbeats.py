"""Turn the assistant's heartbeat runs into labelable cases; reads the transcript store read-only and changes nothing.

One heartbeat poll is one case: the poll, everything the assistant produced before the next
human message, the previous heartbeat's output, whether a human wrote anything in between,
and (best effort) whether a Telegram send followed. Run it from the MacBook; the worker half
executes on the Mac mini over SSH because the SQLite store lives there.

    python3 -m scripts.extract_heartbeats --inspect 5          # look at the stored event shape first
    python3 -m scripts.extract_heartbeats --since 2026-09-21    # write runs/heartbeats-<stamp>/cases.{jsonl,csv}
    python3 -m scripts.extract_heartbeats --local-db copy.db    # same logic against a local copy

Labels go into the CSV's empty columns (decision, poll_treated_as_human, stale_fact, notes);
decision is useful / somewhat_useful / redundant for a sent message, correct_silence / missed for a silent run.
"""

import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

POLL_MARKER = "[OpenClaw heartbeat poll]"
SENTINELS = ("NO_REPLY", "HEARTBEAT_OK")
# A control token emitted as a line of its own (optionally behind a label such as "**Heartbeat Result:**"),
# as opposed to a mention inside a sentence ("the NO_REPLY protocol correction"), which is not a leak.
SENTINEL_LINE = re.compile(r"(?m)^\s*(?:\*\*[^*\n]{0,60}\*\*\s*:?\s*|[A-Za-z ]{0,40}:\s*)?(?:NO_REPLY|HEARTBEAT_OK)\b\s*[.!]?\s*$")
DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"        # gitignored; optional OPENCLAW_SSH_HOST, OPENCLAW_HOME, OPENCLAW_MAIN_SESSION settings
DEFAULT_SEARCH_ROOT = "~/.openclaw"                             # the assistant user's OpenClaw home on the machine that holds the store
DEFAULT_LOG_DIR = "/tmp/openclaw"
SEND_OK_MARKER = "telegram outbound send ok"
DECISIONS = ("useful", "somewhat_useful", "redundant", "correct_silence", "missed")  # the label vocabulary; see README.md
LABEL_COLUMNS = ("decision", "poll_treated_as_human", "stale_fact", "notes")
TEXT_TYPES = {"text", "input_text", "output_text"}
CALL_TYPES = {"toolCall", "tool_call", "tool_use", "function_call"}
GREETING = re.compile(r"^\W*(hi|hello|hey|welcome back|good (morning|afternoon|evening))\b", re.IGNORECASE)
# Crude hints for sorting the CSV by how the user reacted; the label is still the human's call.
REACTION_POSITIVE = re.compile(r"\b(thanks|thank you|great|perfect|nice|helpful|good catch|yes,? (please|do|go)|go ahead|do (that|it)|sounds good)\b", re.IGNORECASE)
REACTION_NEGATIVE = re.compile(r"\b(NO_REPLY|stop|don'?t|do not|shouldn'?t|should not|why (did|are) you|wrong|again\?|not what I)\b", re.IGNORECASE)
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\d-])")


# ----- shared helpers (used on both machines) -----------------------------------------------

def parse_time(value):
    """Accept ISO strings or epoch seconds/milliseconds; return an aware UTC datetime or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{10,13}(\.\d+)?", text):
            return parse_time(float(text))
        try:
            stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return None


def normalize_event(raw, created_at, seq):
    """Map one stored transcript event to a flat row; supports the {type, message:{...}} envelope and flat messages."""
    event = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(event, dict):
        return None
    message = event["message"] if isinstance(event.get("message"), dict) else event
    role = message.get("role") or event.get("role")
    if not isinstance(role, str):
        return None
    content = message.get("content")
    if content is None:
        content = message.get("text", "")
    texts, calls = [], []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind in TEXT_TYPES and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif kind in CALL_TYPES:
                    function = block.get("function") if isinstance(block.get("function"), dict) else {}
                    calls.append(str(block.get("name") or function.get("name") or "unknown"))
    at = (parse_time(message.get("timestamp")) or parse_time(event.get("timestamp"))
          or parse_time(event.get("ts")) or parse_time(created_at))
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else event.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {"seq": seq, "at": at.isoformat() if at else None, "role": role,
            "text": "\n".join(text for text in texts if text),
            "tool_calls": calls,
            "stop_reason": message.get("stopReason") or message.get("stop_reason") or event.get("stopReason"),
            "model": message.get("model") or event.get("model"),
            "usage": {key: value for key, value in usage.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)}}


def describe_shape(raw):
    """Summarize an event's structure without its content, for adapting normalize_event quickly."""
    event = json.loads(raw)
    if not isinstance(event, dict):
        return {"json_type": type(event).__name__}
    message = event["message"] if isinstance(event.get("message"), dict) else None
    content = (message or event).get("content")
    if isinstance(content, list):
        kinds = sorted({block.get("type", "?") if isinstance(block, dict) else type(block).__name__ for block in content})
    else:
        kinds = type(content).__name__
    return {"top_keys": sorted(event.keys())[:25], "message_keys": sorted(message.keys())[:25] if message else None,
            "role": (message or event).get("role"), "content_block_types": kinds,
            "timestamp_fields": [key for key in ("timestamp", "ts", "created_at", "createdAt") if key in event or (message and key in message)]}


# ----- worker (runs on the machine that holds the database) ---------------------------------

def load_env(path):
    """KEY=VALUE lines from .env; variables already set win; comments and blanks are skipped."""
    path = Path(path)
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def connect_readonly(path):
    """Open SQLite strictly read-only; the assistant's store is never written by this tool."""
    connection = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def find_database(explicit, search_root, session):
    """Use the given path, or find the store containing transcript_events (and the session) under the root."""
    candidates = [Path(os.path.expanduser(explicit))] if explicit else []
    if not explicit:
        for directory, subdirectories, files in os.walk(os.path.expanduser(search_root)):
            subdirectories[:] = [name for name in subdirectories if name not in {"node_modules", ".git", "backups"}]
            candidates.extend(Path(directory) / name for name in files if name.endswith((".db", ".sqlite", ".sqlite3")))
    tried = []
    for path in sorted(candidates):
        try:
            connection = connect_readonly(path)
            try:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "transcript_events" not in tables:
                    tried.append(str(path) + ": no transcript_events table")
                    continue
                if session and not connection.execute("SELECT COUNT(*) FROM transcript_events WHERE session_id=?", (session,)).fetchone()[0]:
                    tried.append(str(path) + ": session not present")
                    continue
                return path
            finally:
                connection.close()
        except sqlite3.Error as error:
            tried.append(str(path) + ": " + str(error))
    raise SystemExit("No transcript database found. Tried: " + ("; ".join(tried) or "nothing under " + str(search_root)))


def scan_send_log(log_dir):
    """Collect timestamps of Telegram send-success lines; the logs carry no message body or run ID."""
    times = []
    directory = Path(log_dir)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("openclaw-*.log"))
    if not files:
        return None
    for path in files:
        try:
            with path.open(errors="replace") as handle:
                for line in handle:
                    if SEND_OK_MARKER in line:
                        found = TIMESTAMP.search(line)
                        stamp = parse_time(found.group(0)) if found else None
                        if stamp:
                            times.append(stamp.isoformat())
        except OSError:
            continue
    return times


def worker(params):
    """Read the selected session and return normalized rows plus send-log times; content is redacted locally later."""
    path = find_database(params.get("db"), params.get("search_root") or DEFAULT_SEARCH_ROOT, params.get("session"))
    connection = connect_readonly(path)
    try:
        session, chosen = params.get("session"), "given"
        if not session:  # no session configured: take the one with the most rows, which is the assistant's main session
            row = connection.execute("SELECT session_id, COUNT(*) AS n FROM transcript_events GROUP BY session_id ORDER BY n DESC LIMIT 1").fetchone()
            session, chosen = (row[0] if row else None), "largest"
        query, arguments = "SELECT seq, event_json, created_at FROM transcript_events", ()
        if session:
            query += " WHERE session_id=?"
            arguments = (session,)
        stored = connection.execute(query + " ORDER BY seq", arguments).fetchall()
    finally:
        connection.close()
    result = {"database": str(path), "session": session, "session_chosen": chosen, "stored_rows": len(stored)}
    inspect = params.get("inspect")
    if inspect:
        picked = stored[:inspect] + stored[-inspect:] if len(stored) > 2 * inspect else stored
        result["shapes"] = [{"seq": seq, "created_at": created_at, "shape": describe_shape(raw)} for seq, raw, created_at in picked]
        return result
    rows = []
    for seq, raw, created_at in stored:
        try:
            row = normalize_event(raw, created_at, seq)
        except (ValueError, TypeError):
            row = None
        if row:
            rows.append(row)
    result["rows"] = rows
    result["unparsed_rows"] = len(stored) - len(rows)
    result["send_ok_times"] = scan_send_log(params.get("log_dir") or DEFAULT_LOG_DIR)
    return result


# ----- local side ---------------------------------------------------------------------------

def redact(text):
    """Remove credential patterns, e-mail addresses and phone numbers from assistant text before it leaves the run directory."""
    if not text:
        return text
    from scripts.export_trace import sanitize_value  # local import: the worker half has no repository on the Mac mini
    text = sanitize_value(text)
    text = EMAIL.sub("[EMAIL]", text)
    return PHONE.sub("[PHONE]", text)


DEFAULT_ZONE = "America/Los_Angeles"


def parse_cli_time(value, zone_name=DEFAULT_ZONE):
    """A --since/--until value: an explicit offset or Z is honored; a naive time is read in the user's zone, not UTC."""
    text = value.strip()
    if re.fullmatch(r"\d{10,13}(\.\d+)?", text):
        return parse_time(float(text))
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is not None:
        return stamp
    try:
        from zoneinfo import ZoneInfo
        return stamp.replace(tzinfo=ZoneInfo(zone_name))
    except Exception:  # noqa: BLE001  (no tz database: fall back to UTC and say so in the summary)
        return stamp.replace(tzinfo=timezone.utc)


def local_time(iso):
    """Render an ISO timestamp in America/Los_Angeles when zoneinfo is available."""
    stamp = parse_time(iso)
    if not stamp:
        return None
    try:
        from zoneinfo import ZoneInfo
        return stamp.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:  # noqa: BLE001  (no tz database: keep UTC)
        return stamp.strftime("%Y-%m-%d %H:%M UTC")


def classify_texts(texts):
    """Silence and sentinel checks mirror the runtime: only a standalone sentinel counts as silence."""
    standalone = [text for text in texts if text in SENTINELS]
    substantive = [text for text in texts if text not in SENTINELS]
    runtime_silent = not substantive
    sentinel_leak = any(SENTINEL_LINE.search(text) for text in substantive)
    return runtime_silent, sentinel_leak, len(standalone)


def classify_completion(final_assistant):
    """Classify the final assistant row conservatively without inferring external delivery."""
    if final_assistant is None:
        return "unknown"
    stop = str(final_assistant.get("stop_reason") or "").strip().lower()
    if stop in {"length", "max_tokens"}:
        return "incomplete"
    if stop in {"error", "failed"}:
        return "failed"
    if stop in {"timeout", "timed_out", "timedout"}:
        return "timeout"
    if stop in {"aborted", "cancelled", "canceled"}:
        return "aborted"
    if not final_assistant.get("tool_calls") and stop in {"stop", "end_turn", "completed"}:
        return "complete"
    return "unknown"


def similarity(previous, current):
    """Rough repeat detector on the first 4,000 characters; 1.0 means the message is a copy of the last one."""
    if not previous or not current:
        return None
    return round(difflib.SequenceMatcher(None, previous[:4000], current[:4000]).ratio(), 3)


def delivery_status(row_times, send_ok_times, window_seconds):
    """Best-effort match: a send-success log line shortly after an assistant row. Timing association, not a receipt."""
    if send_ok_times is None:
        return "unknown", 0
    sends = [parse_time(value) for value in send_ok_times]
    matches = 0
    for iso in row_times:
        at = parse_time(iso)
        if at and any(send and 0 <= (send - at).total_seconds() <= window_seconds for send in sends):
            matches += 1
    return ("matched" if matches else "none"), matches


USER_MESSAGE_LIMIT, USER_MESSAGES_LIMIT = 500, 2000


def join_user_messages(rows):
    """What the user wrote between two polls, as the judge and the labeler both see it; bounded so one long paste cannot dominate."""
    parts = []
    for row in rows:
        text = " ".join((row.get("text") or "").split())
        if text:
            parts.append("[user] " + text[:USER_MESSAGE_LIMIT])
    joined = "\n".join(parts)
    return joined[:USER_MESSAGES_LIMIT] + ("\n[truncated]" if len(joined) > USER_MESSAGES_LIMIT else "")


def reaction_hint(text):
    """positive / negative / neutral from keyword patterns; a sorting aid for the labeler, never a label."""
    if not text:
        return None
    if REACTION_NEGATIVE.search(text):
        return "negative"
    if REACTION_POSITIVE.search(text):
        return "positive"
    return "neutral"


def assign_split(case_id):
    """Deterministic tune/holdout halves so the judge is tuned and reported on different cases."""
    return "tune" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def build_cases(rows, session, send_ok_times=None, delivery_window_seconds=3.0, since=None, until=None):
    """Pair each heartbeat poll with what followed it and with the previous heartbeat's output."""
    rows = sorted((row for row in rows if row and row.get("role")), key=lambda row: row["seq"])
    polls = [index for index, row in enumerate(rows) if row["role"] == "user" and POLL_MARKER in (row.get("text") or "")]
    cases, previous = [], None
    for number, index in enumerate(polls):
        poll = rows[index]
        end = next((j for j in range(index + 1, len(rows)) if rows[j]["role"] == "user"), len(rows))
        window = rows[index + 1:end]
        assistant_rows = [row for row in window if row["role"] == "assistant"]
        final_assistant = assistant_rows[-1] if assistant_rows else None
        texts = [row["text"].strip() for row in assistant_rows if (row.get("text") or "").strip()]
        runtime_silent, sentinel_leak, standalone = classify_texts(texts)
        substantive = [text for text in texts if text not in SENTINELS]
        output = "\n\n".join(substantive)
        start = polls[number - 1] + 1 if number else 0
        between = rows[start:index]
        human = [row for row in between if row["role"] == "user" and POLL_MARKER not in (row.get("text") or "")]
        poll_at, last_at = parse_time(poll.get("at")), parse_time(window[-1]["at"]) if window else None
        previous_poll_at = parse_time(rows[polls[number - 1]].get("at")) if number else None
        usage_input = [row["usage"].get("input") for row in assistant_rows if row.get("usage", {}).get("input") is not None]
        usage_output = [row["usage"].get("output") for row in assistant_rows if row.get("usage", {}).get("output") is not None]
        row_times = [row["at"] for row in assistant_rows if (row.get("text") or "").strip() and row.get("at")]
        delivery, delivery_matches = delivery_status(row_times, send_ok_times, delivery_window_seconds)
        # The user's first message after this heartbeat, if it is not just the next poll: the labeler's best evidence.
        reaction = rows[end] if end < len(rows) and POLL_MARKER not in (rows[end].get("text") or "") else None
        reaction_at = parse_time(reaction.get("at")) if reaction else None
        replied_at = last_at or poll_at
        case_id = "hb-" + str(poll["seq"])
        case = {
            "case_id": case_id, "session": session, "seq": poll["seq"], "at": poll.get("at"), "at_local": local_time(poll.get("at")),
            "trigger": "heartbeat poll", "poll_text": poll.get("text"),
            "minutes_since_prev": round((poll_at - previous_poll_at).total_seconds() / 60, 1) if poll_at and previous_poll_at else None,
            "user_activity_since_prev": bool(human), "human_messages_between": len(human),
            "user_messages_text": join_user_messages(human),
            "prev_case_id": previous["case_id"] if previous else None,
            "prev_output": previous["output"] if previous else "",
            "runtime_silent": runtime_silent, "sentinel_leak": sentinel_leak, "standalone_sentinels": standalone,
            "greeting_detected": bool(output) and bool(GREETING.match(output)),
            "similarity_to_prev": similarity(previous["output"] if previous else "", output),
            "output": output, "final_text": substantive[-1] if substantive else "", "output_chars": len(output),
            "completion": classify_completion(final_assistant),
            "final_response": final_assistant.get("text") if final_assistant is not None else None,
            "assistant_text_rows": len(texts), "assistant_rows": len(assistant_rows),
            "tool_calls": [name for row in window for name in row.get("tool_calls", [])],
            "duration_s": round((last_at - poll_at).total_seconds(), 1) if poll_at and last_at else None,
            "input_tokens_max": max(usage_input) if usage_input else None,
            "output_tokens_sum": sum(usage_output) if usage_output else None,
            "delivery": delivery, "delivery_matches": delivery_matches,
            "reaction_text": (reaction.get("text") or "") if reaction else "",
            "reaction_seq": reaction["seq"] if reaction else None,
            "reaction_minutes": round((reaction_at - replied_at).total_seconds() / 60, 1) if reaction_at and replied_at else None,
            "reaction_hint": reaction_hint((reaction.get("text") or "") if reaction else ""),
            "split": "silent" if runtime_silent else assign_split(case_id),
        }
        previous = case
        if since and poll_at and poll_at < since:
            continue
        if until and poll_at and poll_at > until:
            continue
        cases.append(case)
    return cases


def summarize(cases):
    """Counts a reviewer can quote without reading the cases."""
    with_output = [case for case in cases if not case["runtime_silent"]]
    durations = [case["duration_s"] for case in cases if case["duration_s"] is not None]
    contexts = [case["input_tokens_max"] for case in cases if case["input_tokens_max"] is not None]
    return {
        "cases": len(cases), "runtime_silent": len(cases) - len(with_output), "with_output": len(with_output),
        "sentinel_leaks": sum(case["sentinel_leak"] for case in cases),
        "greetings_after_poll": sum(case["greeting_detected"] for case in cases),
        "near_repeats_over_0_8": sum(1 for case in with_output if (case["similarity_to_prev"] or 0) > 0.8),
        "with_output_and_no_user_activity": sum(1 for case in with_output if not case["user_activity_since_prev"]),
        "delivery": {status: sum(1 for case in cases if case["delivery"] == status) for status in ("matched", "none", "unknown")},
        "reactions": {hint: sum(1 for case in cases if case["reaction_hint"] == hint) for hint in ("positive", "negative", "neutral")},
        "output_chars_total": sum(case["output_chars"] for case in cases),
        "duration_s_median": sorted(durations)[len(durations) // 2] if durations else None,
        "input_tokens_max_median": sorted(contexts)[len(contexts) // 2] if contexts else None,
        "splits": {split: sum(1 for case in cases if case["split"] == split) for split in ("tune", "holdout", "silent")},
    }


CSV_COLUMNS = ("case_id", "at_local", "minutes_since_prev", "user_activity_since_prev", "runtime_silent", "sentinel_leak",
               "greeting_detected", "similarity_to_prev", "delivery", "duration_s", "input_tokens_max", "output_tokens_sum",
               "output_chars", "split", "user_messages_preview", "reaction_hint", "reaction_minutes", "reaction_preview", "prev_output_preview", "output") + LABEL_COLUMNS


def write_outputs(cases, directory, summary):
    """Write the full JSONL, a labelable CSV and the summary; nothing is exported anywhere."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    with (directory / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in cases:
            row = {column: case.get(column) for column in CSV_COLUMNS if column in case}
            row["prev_output_preview"] = " ".join((case["prev_output"] or "")[:500].split())
            row["reaction_preview"] = " ".join((case["reaction_text"] or "")[:300].split())
            row["user_messages_preview"] = " ".join((case["user_messages_text"] or "")[:300].split())
            row["output"] = (case["output"] or "")[:8000]
            row.update({column: "" for column in LABEL_COLUMNS})
            writer.writerow(row)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in directory.iterdir():
        os.chmod(path, 0o600)


def run_worker_remotely(host, params):
    """Ship this file to the Mac mini and run its worker half there; only JSON comes back."""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "python3 - " + shlex.quote(json.dumps(params))]
    result = subprocess.run(command, input=Path(__file__).read_text(), text=True, capture_output=True, timeout=300)
    if result.returncode:
        raise SystemExit("Remote worker failed (exit " + str(result.returncode) + "):\n" + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    load_env(DEFAULT_ENV)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("OPENCLAW_SSH_HOST", ""), help="user@host that holds the transcript store (default OPENCLAW_SSH_HOST from .env)")
    parser.add_argument("--local-db", type=Path, help="Run against a local database copy instead of SSH")
    parser.add_argument("--db", help="Explicit database path on the target; otherwise searched under --search-root")
    parser.add_argument("--search-root", default=os.environ.get("OPENCLAW_HOME", DEFAULT_SEARCH_ROOT), help="OpenClaw home on the target (default OPENCLAW_HOME from .env)")
    parser.add_argument("--session", default=os.environ.get("OPENCLAW_MAIN_SESSION", ""), help="Session ID to read (default OPENCLAW_MAIN_SESSION from .env); empty picks the largest session")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Gateway log directory on the target for send-ok matching")
    parser.add_argument("--delivery-window", type=float, default=3.0, help="Seconds after an assistant row to accept a send-ok line")
    parser.add_argument("--since", help="Keep polls at or after this date/time (ISO; read as --tz unless an offset or Z is given)")
    parser.add_argument("--until", help="Keep polls at or before this date/time (same rule)")
    parser.add_argument("--tz", default=DEFAULT_ZONE, help="Zone for --since/--until values without an offset (default America/Los_Angeles)")
    parser.add_argument("--inspect", type=int, metavar="N", help="Print the stored event shape for the first and last N rows and stop")
    parser.add_argument("--out", type=Path, help="Output directory (default runs/heartbeats-<stamp>)")
    parser.add_argument("--no-redact", action="store_true", help="Keep e-mail addresses and phone numbers (credential patterns are always removed)")
    args = parser.parse_args()
    params = {"db": args.db, "search_root": args.search_root, "session": args.session or None,
              "log_dir": args.log_dir, "inspect": args.inspect}
    if not args.local_db and not args.host:
        parser.error("Set OPENCLAW_SSH_HOST in .env or pass --host (or use --local-db)")
    if args.local_db:
        params["db"] = str(args.local_db)
        params["search_root"] = str(args.local_db.parent)
        result = worker(params)
    else:
        result = run_worker_remotely(args.host, params)
    if args.inspect:
        print(json.dumps({key: result[key] for key in ("database", "session", "stored_rows", "shapes")}, indent=2))
        return 0
    since = parse_cli_time(args.since, args.tz) if args.since else None
    until = parse_cli_time(args.until, args.tz) if args.until else None
    if (args.since and not since) or (args.until and not until):
        parser.error("--since/--until must be ISO timestamps, e.g. 2026-09-21 or 2026-09-28T22:00 (read as --tz) or 2026-09-29T05:00:00Z")
    rows = result["rows"]
    if not args.no_redact:
        for row in rows:
            row["text"] = redact(row.get("text") or "")
    cases = build_cases(rows, result.get("session"), result.get("send_ok_times"), args.delivery_window, since, until)
    summary = {"database": result["database"], "session": result.get("session"), "session_chosen": result.get("session_chosen"), "stored_rows": result["stored_rows"],
               "unparsed_rows": result.get("unparsed_rows", 0), "send_ok_lines": len(result.get("send_ok_times") or []),
               "send_log_available": result.get("send_ok_times") is not None, "redacted": not args.no_redact,
               "since": since.isoformat() if since else None, "until": until.isoformat() if until else None, **summarize(cases)}
    directory = args.out or Path(__file__).resolve().parents[1] / "runs" / ("heartbeats-" + datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz"))
    write_outputs(cases, directory, summary)
    print(json.dumps({**summary, "run_dir": str(directory)}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lstrip().startswith("{"):
        print(json.dumps(worker(json.loads(sys.argv[1]))))
    else:
        raise SystemExit(main())
