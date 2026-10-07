"""Heartbeat judge in Phoenix: upload labeled cases as a dataset, run the judge as an experiment, score it against the labels.

The task under test is the judge, not the assistant: the experiment measures how often an LLM
that sees preserved messages and scenario evidence agrees with human reference labels.
Completed silence is judged as correct_silence, missed, or unsure; missing evidence is not success.
Freeze the calibrated judge before comparing assistant behavior before and after a change.

    # 1. iterate on the prompt without Phoenix (prints judge JSON next to the human label)
    python3 -m evals.judge_heartbeats try --cases runs/RUN/cases.jsonl --labels runs/RUN/cases.csv --limit 4

    # 2. upload the labeled cases (tune/holdout/silent splits come from the extractor)
    python3 -m evals.judge_heartbeats build-dataset --cases runs/RUN/cases.jsonl \
        --labels runs/RUN/cases.csv --name openclaw-heartbeats-v1 --period pre

    # 3. run the judge as an experiment on one split
    python3 -m evals.judge_heartbeats run --name openclaw-heartbeats-v1 --split holdout \
        --experiment judge-v1-holdout --reps 1

    # 4. after editing labels or notes in the Phoenix UI, pull them back into the CSV
    python3 -m evals.judge_heartbeats pull-labels --name openclaw-heartbeats-v1 --merge-into runs/RUN/cases.csv

Credentials come from ./.env (see .env.example: PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_API_KEY and the
judge endpoint key) unless already set in the environment. Dependencies are in requirements.txt.
The judge defaults to the Qwen endpoint on spark with thinking off; pass --judge-base-url,
--judge-model and --judge-api-key-env (or the HEARTBEAT_JUDGE_* variables) for anything else.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from scripts.extract_heartbeats import SENTINEL_LINE, SENTINELS

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "heartbeat-judge.md"
DEFAULT_ENV = ROOT / ".env"                                    # gitignored; .env.example lists the keys
DEFAULT_JUDGE_URL = "http://dgx-spark:8002/v1"                 # the vLLM server on spark; any OpenAI-compatible endpoint works
DEFAULT_JUDGE_MODEL = "qwen3.5-122b-a10b"
DEFAULT_JUDGE_KEY_ENV = "SPARK_API_KEY"
LEVELS = ("useful", "somewhat_useful", "redundant")            # what the judge (and the human) says about a sent message
USEFULNESS_SCORE = {"useful": 1.0, "somewhat_useful": 0.5, "redundant": 0.0}
DECISIONS = LEVELS + ("correct_silence", "missed")             # human labels; the last two apply to silent runs only
JUDGE_DECISIONS = DECISIONS + ("unsure",)
JUDGE_OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"usefulness": {"type": "string", "enum": list(JUDGE_DECISIONS)},
                   "new_information": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["usefulness", "new_information", "explanation"],
}
PREV_LIMIT, OUTPUT_LIMIT, USER_LIMIT = 3000, 6000, 2000
CONTEXT_FIELDS = ("trigger", "clock", "completion", "user_intent", "prior_delivery", "evidence", "scope",
                  "response_contract", "evidence_coverage", "prior_notifications", "current_run_outbound_messages")
TRUE_WORDS, FALSE_WORDS = {"1", "true", "yes", "y", "x"}, {"0", "false", "no", "n", ""}


# ----- configuration --------------------------------------------------------------------------

def load_env(path):
    """Read KEY=VALUE lines without overriding variables already set; comments and blanks are skipped."""
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


def read_prompt(path=PROMPT_PATH):
    """Read the judge rubric and return its short content fingerprint."""
    template = Path(path).read_text()
    return template, hashlib.sha256(template.encode()).hexdigest()[:12]


def build_judge_context(case_input):
    """Select scenario evidence without including reference labels or reviewer notes."""
    context = case_input.get("judge_context") or {}
    return {key: context[key] for key in CONTEXT_FIELDS if key in context}


def classify_response(case_input):
    """Distinguish recorded message content, demonstrated final suppression, and unknown output."""
    context = build_judge_context(case_input)
    outbound = context.get("current_run_outbound_messages") or []
    if any(message.get("text", "").strip() for message in outbound):
        return "message", "Recorded outbound message content is available."
    output = (case_input.get("output") or "").strip()
    final = case_input.get("final_response", case_input.get("final_text"))
    final_suppression = bool(final and final.strip() in SENTINELS)
    if not final_suppression and any(text and text.strip() not in SENTINELS for text in (output, final)):
        return "message", "Substantive assistant text is available; delivery is a separate question."
    completion = case_input.get("completion") or context.get("completion")
    if completion not in {"complete", "completed"}:
        return "unknown", "Run completion is " + str(completion or "unknown") + "; no deliberate silence is established."
    if not final or final.strip() not in SENTINELS:
        return "unknown", "No completed standalone suppression token was preserved."
    if output and output not in SENTINELS:
        return "unknown", "A final suppression token follows intermediate text whose delivery is unknown."
    # Names alone cannot establish what a messaging tool sent or whether it succeeded.
    if any(name == "message" or "send" in name.lower() for name in case_input.get("tool_calls", [])):
        return "unknown", "A messaging tool may have sent content; its message/result evidence is missing."
    return "silent", "A completed final suppression token was observed; actual delivery remains separate."


def render_prompt(template, case_input):
    """Substitute context once so placeholder-like text inside evidence remains literal data."""
    context = build_judge_context(case_input)
    values = {"{minutes}": str(case_input.get("minutes_since_prev", "unknown")),
              "{user_activity}": "yes" if case_input.get("user_activity_since_prev") else "no",
              "{user_messages}": (case_input.get("user_messages_between") or case_input.get("user_messages_text") or "")[:USER_LIMIT] or "(none)",
              "{prev_output}": (case_input.get("prev_output") or "")[:PREV_LIMIT] or "(empty)",
              "{output}": (case_input.get("output") or "")[:OUTPUT_LIMIT] or "(empty)",
              "{trigger}": str(context.get("trigger") or case_input.get("trigger") or "unknown"),
              "{completion}": str(case_input.get("completion") or context.get("completion") or "unknown"),
              "{response_state}": classify_response(case_input)[0],
              "{final_response}": (case_input.get("final_response", case_input.get("final_text")) or "(not recorded)")[:OUTPUT_LIMIT],
              "{judge_context}": json.dumps(context, ensure_ascii=False, indent=2)}
    return re.sub("|".join(re.escape(key) for key in values), lambda match: values[match.group()], template)


def as_bool(value):
    """CSV cells arrive as text; accept the usual spellings and return None when the cell is blank."""
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return None if text == "" else False
    raise ValueError("Unrecognized boolean cell: " + repr(value))


def normalize_level(value):
    """Accept 'somewhat useful', 'Somewhat-Useful' and friends."""
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


def expected_level(human):
    """Map a human label to the verdict the judge should give, or None when the judge cannot be scored on it."""
    return human if human in JUDGE_DECISIONS else None


# ----- the judge --------------------------------------------------------------------------------

def parse_judge_json(text):
    """Extract the first JSON object from a model reply, tolerating code fences, thinking blocks and stray prose."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty judge reply")
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object in judge reply")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:].strip())
    if not isinstance(parsed, dict):
        raise ValueError("Judge reply is not a JSON object")
    level = normalize_level(parsed.get("usefulness"))
    if level not in JUDGE_DECISIONS:
        raise ValueError("Judge usefulness must be one of " + ", ".join(JUDGE_DECISIONS) + "; got " + repr(parsed.get("usefulness")))
    result = {"usefulness": level, "score": USEFULNESS_SCORE.get(level),
              "new_information": str(parsed.get("new_information", "")), "explanation": str(parsed.get("explanation", ""))}
    if "poll_treated_as_human" in parsed:
        result["poll_treated_as_human"] = as_bool(parsed["poll_treated_as_human"])
    return result


def make_judge(base_url, model, api_key, thinking=False, temperature=0.0, max_tokens=None,
               provider="vllm", reasoning_effort="low"):
    """Build a provider-specific text judge that preserves usage and response completion metadata."""
    if provider not in {"vllm", "openai"}:
        raise ValueError("Judge provider must be vllm or openai")
    if provider == "openai" and not api_key:
        raise ValueError("The OpenAI judge requires a configured API key")
    if provider == "openai" and thinking:
        raise ValueError("Use --reasoning-effort for OpenAI; --thinking is a vLLM option")
    if provider == "openai" and model.startswith("gpt-6-astra") and reasoning_effort == "none":
        raise ValueError("GPT-6 Astra requires reasoning effort low or higher")
    limit = max_tokens if max_tokens is not None else (4096 if provider == "openai" else 600)
    if limit <= 0:
        raise ValueError("Judge token limit must be positive")
    from openai import OpenAI  # requirements.txt; imported lazily so the tests need no network client

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=120, max_retries=0)
    if provider == "openai":
        parameters = {"max_completion_tokens": limit, "reasoning_effort": reasoning_effort,
                      "response_format": {"type": "json_schema", "json_schema": {
                          "name": "heartbeat_usefulness", "strict": True, "schema": JUDGE_OUTPUT_SCHEMA}}}
    else:
        parameters = {"temperature": temperature, "max_tokens": limit,
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}

    def call(prompt):
        """Submit one judgment and retain tokens, latency, model identity, and completion status."""
        start = time.monotonic()
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **parameters)
        choice = response.choices[0] if response.choices else None
        return {"text": (choice.message.content or "") if choice else "",
                "model_usage": {"requested_model": model, "returned_model": response.model,
                                "usage": response.usage.model_dump() if response.usage else None,
                                "latency_seconds": round(time.monotonic() - start, 3),
                                "finish_reason": choice.finish_reason if choice else None,
                                "refusal": bool(getattr(choice.message, "refusal", None)) if choice else False,
                                "reasoning_effort": reasoning_effort if provider == "openai" else None,
                                "max_completion_tokens": limit}}
    return call


def judge_case(case_input, call, template):
    """Judge messages and evidenced silence; preserve uncertainty when response evidence is missing."""
    state, reason = classify_response(case_input)
    context = build_judge_context(case_input)
    if state == "silent" and not (context.get("user_intent") and context.get("evidence")):
        state, reason = "unknown", "Completed suppression is recorded, but task/notification evidence is missing."
    if state == "unknown":
        return {"usefulness": "unsure", "score": None, "new_information": "unknown",
                "explanation": reason, "judged_by": "rule", "response_state": state}
    reply = call(render_prompt(template, case_input))
    # Offline stubs and existing integrations can still supply plain response text.
    raw = reply["text"] if isinstance(reply, dict) else reply
    metadata = {"model_usage": reply["model_usage"]} if isinstance(reply, dict) else {}
    try:
        if metadata and (metadata["model_usage"].get("finish_reason") != "stop" or metadata["model_usage"].get("refusal")):
            raise ValueError("Judge response did not complete normally or was refused; do not score it as a valid judgment")
        parsed = parse_judge_json(raw)
        allowed = ("correct_silence", "missed", "unsure") if state == "silent" else LEVELS + ("unsure",)
        if parsed["usefulness"] not in allowed:
            raise ValueError("Verdict " + parsed["usefulness"] + " is incompatible with response state " + state)
    except ValueError as error:
        return {"usefulness": "unparsed", "score": None, "new_information": "",
                "explanation": str(error), "judged_by": "model", "raw": raw[:2000], "response_state": state, **metadata}
    return {**parsed, "judged_by": "model", "raw": raw[:2000], "response_state": state, **metadata}


# ----- evaluators (input, output, expected are bound by name by the Phoenix client) -----------

def usefulness_agreement(input, output, expected):
    """Score exact agreement on messages and silence, keeping abstentions in the labeled denominator."""
    human = (expected or {}).get("decision")
    got = (output or {}).get("usefulness")
    want = expected_level(human)
    if want is None:
        return {"label": "unlabeled", "explanation": "No human usefulness label on this example."}
    agree = got == want
    return {"score": float(agree), "label": "agree" if agree else "disagree", "explanation": "judge=" + str(got) + ", human=" + str(human) + "."}


def usefulness_distance(input, output, expected):
    """1 minus the gap on the 1 / 0.5 / 0 scale: one level off scores 0.5, two levels off scores 0."""
    human = (expected or {}).get("decision")
    got = (output or {}).get("usefulness")
    if human not in LEVELS or got not in LEVELS:
        return {"label": "not comparable", "explanation": "Needs a human level and a judge level for a sent message; human=" + str(human) + ", judge=" + str(got) + "."}
    gap = abs(USEFULNESS_SCORE[got] - USEFULNESS_SCORE[human])
    return {"score": 1.0 - gap, "label": "exact" if gap == 0 else "one level off" if gap == 0.5 else "two levels off",
            "explanation": "judge=" + got + ", human=" + human + "."}


def redundant_agreement(input, output, expected):
    """Collapsed to the production question, did this message need sending: redundant versus anything useful."""
    human = (expected or {}).get("decision")
    got = (output or {}).get("usefulness")
    if human not in LEVELS or got not in LEVELS:
        return {"label": "not comparable", "explanation": "Needs a human level and a judge level for a sent message."}
    agree = (got == "redundant") == (human == "redundant")
    return {"score": float(agree), "label": "agree" if agree else "disagree", "explanation": "judge=" + got + ", human=" + human + "."}


def poll_treated_as_human_agreement(input, output, expected):
    """Compare optional legacy timer-confusion annotations only when both sides supplied one."""
    try:
        want = as_bool((expected or {}).get("poll_treated_as_human"))
    except ValueError:
        want = None
    if want is None:
        return {"label": "unlabeled", "explanation": "No human poll_treated_as_human label."}
    got = as_bool((output or {}).get("poll_treated_as_human"))
    if got is None:
        return {"label": "not assessed", "explanation": "The judge did not return this optional diagnostic."}
    return {"score": float(got == want), "label": "agree" if got == want else "disagree", "explanation": "judge=" + str(got) + ", human=" + str(want) + "."}


def judge_usefulness(input, output, expected):
    """The judge's own verdict as a number, so an experiment's mean is mean usefulness and its 'redundant' share is the redundant rate."""
    got = (output or {}).get("usefulness")
    if got not in LEVELS:
        return {"label": str(got or "none"), "explanation": "Silence, uncertainty, and parse failures are reported separately from mean message usefulness."}
    return {"score": USEFULNESS_SCORE[got], "label": got, "explanation": str((output or {}).get("new_information") or "")[:300]}


def assistant_sentinel_leak(input, output, expected):
    """Check recorded response syntax without rewarding missing output or inferring delivery."""
    case_input = input or {}
    state, reason = classify_response(case_input)
    if state == "unknown":
        return {"label": "not assessed", "explanation": reason}
    if state == "silent":
        return {"score": 1.0, "label": "suppression token", "explanation": "Completed standalone token; this checks syntax, not whether silence was appropriate or delivery occurred."}
    texts = [case_input.get("output") or "", case_input.get("final_response") or ""]
    texts.extend(message.get("text") or "" for message in build_judge_context(case_input).get("current_run_outbound_messages", []))
    leak = any(text.strip() not in SENTINELS and SENTINEL_LINE.search(text) for text in texts)
    return {"score": 0.0 if leak else 1.0, "label": "leak" if leak else "clean",
            "explanation": "A control token stands as a line of its own inside recorded substantive text; delivery is not inferred." if leak else "No control token line in recorded substantive text."}


def assistant_repeat(input, output, expected):
    """Property of the assistant's message: how much of it is a copy of the previous heartbeat (0 to 1)."""
    similarity = (input or {}).get("similarity_to_prev")
    if similarity is None:
        return {"label": "no previous", "explanation": "No previous heartbeat text to compare against."}
    return {"score": float(similarity), "label": "repeat" if similarity > 0.8 else "changed",
            "explanation": "SequenceMatcher ratio against the previous heartbeat: " + str(similarity) + "."}


EVALUATORS = [usefulness_agreement, usefulness_distance, redundant_agreement,
              judge_usefulness, assistant_sentinel_leak, assistant_repeat]


# ----- dataset ----------------------------------------------------------------------------------

def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_labels(path):
    """Labels typed into the extractor's CSV; unknown values are reported, not silently dropped."""
    labels, problems = {}, []
    if not path:
        return labels, problems
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            decision = normalize_level(row.get("decision"))
            if decision and decision not in JUDGE_DECISIONS:
                problems.append(str(row.get("case_id")) + ": unknown decision " + repr(row.get("decision")))
                continue
            try:
                flags = {key: as_bool(row.get(key)) for key in ("poll_treated_as_human", "stale_fact")}
            except ValueError as error:
                problems.append(str(row.get("case_id")) + ": " + str(error))
                continue
            labels[row["case_id"]] = {"decision": decision or None, "notes": (row.get("notes") or "").strip(), **flags}
    return labels, problems


def example_from_case(case, label, period):
    """One Phoenix example: what the judge may see as input, the human label as expected output, the rest as metadata."""
    decision = (label or {}).get("decision")
    expected = {}
    if decision:
        expected = {"decision": decision, "usefulness_score": USEFULNESS_SCORE.get(decision),
                    "poll_treated_as_human": label.get("poll_treated_as_human"), "stale_fact": label.get("stale_fact"), "notes": label.get("notes") or ""}
    return {
        "input": {"case_id": case["case_id"], "at_local": case.get("at_local"), "trigger": case.get("trigger", "heartbeat poll"),
                  "minutes_since_prev": case.get("minutes_since_prev"), "user_activity_since_prev": case.get("user_activity_since_prev"),
                  "user_messages_between": (case.get("user_messages_text") or "")[:USER_LIMIT],
                  "runtime_silent": case.get("runtime_silent"), "similarity_to_prev": case.get("similarity_to_prev"),
                  "prev_output": (case.get("prev_output") or "")[:PREV_LIMIT], "output": (case.get("output") or "")[:OUTPUT_LIMIT],
                  "final_response": case.get("final_response", case.get("final_text")),
                  "completion": case.get("completion") or build_judge_context(case).get("completion") or "unknown",
                  "tool_calls": case.get("tool_calls", []), "judge_context": build_judge_context(case)},
        "output": expected,
        "metadata": {"period": period, "session": case.get("session"), "seq": case.get("seq"), "split": case.get("split"),
                     "delivery": case.get("delivery"), "sentinel_leak": case.get("sentinel_leak"), "greeting_detected": case.get("greeting_detected"),
                     "output_chars": case.get("output_chars"), "input_tokens_max": case.get("input_tokens_max"),
                     "duration_s": case.get("duration_s"), "labeled": bool(decision)},
        "splits": case.get("split") or "unsplit",
    }


def build_examples(cases, labels, period, include_unlabeled=False):
    examples = []
    for case in cases:
        label = labels.get(case["case_id"])
        if not include_unlabeled and not (label and label.get("decision")):
            continue
        examples.append(example_from_case(case, label, period))
    return examples


# ----- commands ---------------------------------------------------------------------------------

def judge_from_args(args):
    """Create the selected provider adapter from explicit model and credential settings."""
    api_key = os.environ.get(args.judge_api_key_env) if args.judge_api_key_env else None
    return make_judge(args.judge_base_url, args.judge_model, api_key, thinking=args.thinking, max_tokens=args.max_tokens,
                      provider=args.judge_provider, reasoning_effort=args.reasoning_effort)


def cmd_try(args):
    """Run the judge on a few labeled cases and print its JSON next to the human label; no Phoenix involved."""
    template, prompt_sha = read_prompt(args.prompt)
    labels, problems = read_labels(args.labels)
    for problem in problems:
        print("label problem:", problem, file=sys.stderr)
    cases = [case for case in read_cases(args.cases) if not labels or labels.get(case["case_id"], {}).get("decision")]
    call = judge_from_args(args)
    exact, distance, counts, confusion = [], [], {}, {}
    for case in cases[:args.limit]:
        example = example_from_case(case, labels.get(case["case_id"]), "try")
        verdict = judge_case(example["input"], call, template)
        human = example["output"].get("decision")
        if human in JUDGE_DECISIONS:
            exact.append(verdict["usefulness"] == human)
            row = confusion.setdefault(human, {})
            row[verdict["usefulness"]] = row.get(verdict["usefulness"], 0) + 1
        if human in LEVELS and verdict["usefulness"] in LEVELS:
            distance.append(1.0 - abs(USEFULNESS_SCORE[verdict["usefulness"]] - USEFULNESS_SCORE[human]))
        counts[verdict["usefulness"]] = counts.get(verdict["usefulness"], 0) + 1
        print(json.dumps({"case_id": case["case_id"], "human": human, "judge": verdict["usefulness"],
                          "new_information": verdict.get("new_information"),
                          "explanation": verdict.get("explanation"), "model_usage": verdict.get("model_usage")}, indent=2))
    print(json.dumps({"prompt_sha256": prompt_sha, "judged": min(len(cases), args.limit),
                      "exact_agreement": (sum(exact) / len(exact)) if exact else None,
                      "mean_distance_score": (sum(distance) / len(distance)) if distance else None,
                      "agreement_cases": len(exact), "message_distance_cases": len(distance),
                      "verdict_counts": counts, "confusion_matrix": confusion}))
    return 0


def cmd_build_dataset(args):
    """Upload the labeled cases (or all cases with --include-unlabeled) as one Phoenix dataset."""
    labels, problems = read_labels(args.labels)
    for problem in problems:
        print("label problem:", problem, file=sys.stderr)
    examples = build_examples(read_cases(args.cases), labels, args.period, args.include_unlabeled)
    if not examples:
        raise SystemExit("No examples to upload: fill the decision column in the CSV or pass --include-unlabeled")
    counts = {}
    for example in examples:
        key = example["output"].get("decision") or "unlabeled"
        counts[key] = counts.get(key, 0) + 1
    if args.dry_run:
        print(json.dumps({"would_upload": len(examples), "decisions": counts, "splits": sorted({example["splits"] for example in examples})}, indent=2))
        return 0
    from phoenix.client import Client

    dataset = Client().datasets.create_dataset(
        name=args.name,
        dataset_description="OpenClaw assistant heartbeat runs (" + args.period + " period) with the user's own usefulness labels; judge alignment and before/after scoring.",
        examples=examples)
    print(json.dumps({"dataset": args.name, "uploaded": len(dataset), "decisions": counts}, indent=2))
    return 0


def bool_cell(value):
    """Render a stored boolean back into the CSV vocabulary."""
    return "" if value is None else ("1" if value else "0")


def labels_from_examples(examples):
    """Read the label fields back out of dataset examples (whatever version the caller fetched)."""
    rows = []
    for example in examples:
        inp, out, meta = example.get("input") or {}, example.get("output") or {}, example.get("metadata") or {}
        rows.append({"case_id": inp.get("case_id"), "decision": out.get("decision") or "", "poll_treated_as_human": bool_cell(out.get("poll_treated_as_human")),
                     "stale_fact": bool_cell(out.get("stale_fact")), "notes": out.get("notes") or "", "split": meta.get("split") or "",
                     "example_id": example.get("id") or "", "updated_at": example.get("updated_at") or ""})
    return rows


def merge_labels_into_csv(path, rows):
    """Overwrite only the label columns of the extractor's CSV, matched by case_id; other columns and rows are kept."""
    by_case = {row["case_id"]: row for row in rows if row.get("case_id")}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, existing = reader.fieldnames, list(reader)
    changed = 0
    for row in existing:
        pulled = by_case.get(row.get("case_id"))
        if not pulled:
            continue
        for column in ("decision", "poll_treated_as_human", "stale_fact", "notes"):
            if column in row and row[column] != pulled[column]:
                row[column] = pulled[column]
                changed += 1
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    return changed, len(existing) - sum(1 for row in existing if row.get("case_id") in by_case)


def cmd_pull_labels(args):
    """Fetch the dataset's current version and write its labels and notes to a CSV, optionally merging into cases.csv."""
    from phoenix.client import Client

    dataset = Client().datasets.get_dataset(dataset=args.name, version_id=args.version_id)
    rows = labels_from_examples(dataset.examples)
    out = args.out or (Path(args.merge_into).parent if args.merge_into else ROOT / "runs") / ("labels-" + args.name + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "decision", "poll_treated_as_human", "stale_fact", "notes", "split", "example_id", "updated_at"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"dataset": args.name, "version_id": dataset.version_id, "examples": len(rows),
               "labeled": sum(1 for row in rows if row["decision"]), "with_notes": sum(1 for row in rows if row["notes"]), "written": str(out)}
    if args.merge_into:
        changed, unmatched = merge_labels_into_csv(args.merge_into, rows)
        summary.update(merged_into=str(args.merge_into), cells_changed=changed, csv_rows_not_in_dataset=unmatched)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run(args):
    """Run the judge over one split as a Phoenix experiment; the evaluators compare it with the human labels."""
    from phoenix.client import Client
    from phoenix.client.experiments import run_experiment

    template, prompt_sha = read_prompt(args.prompt)
    provider = None
    if args.trace_project:
        from phoenix.otel import register
        provider = register(project_name=args.trace_project, auto_instrument=True, batch=True, verbose=False)
    client = Client()
    dataset = client.datasets.get_dataset(dataset=args.name, splits=None if args.split == "all" else [args.split])
    call = judge_from_args(args)
    verdicts = []

    def task(input):
        """Evaluate one dataset example and retain its verdict for the run summary."""
        verdict = judge_case(input, call, template)
        verdicts.append(verdict["usefulness"])
        return verdict

    run_experiment(
        dataset=dataset, task=task, evaluators=EVALUATORS,
        experiment_name=args.experiment, experiment_description="heartbeat judge on split " + args.split + " (prompt " + prompt_sha + ")",
        experiment_metadata={"judge_model": args.judge_model, "judge_base_url": args.judge_base_url, "thinking": args.thinking,
                             "judge_provider": args.judge_provider, "reasoning_effort": args.reasoning_effort if args.judge_provider == "openai" else None,
                             "max_completion_tokens": args.max_tokens if args.max_tokens is not None else (4096 if args.judge_provider == "openai" else 600),
                             "prompt_sha256": prompt_sha, "split": args.split, "dataset": args.name},
        repetitions=args.reps, timeout=args.timeout, dry_run=args.dry_run or False)  # sync runner: sequential, which the local judge endpoint prefers
    if provider:
        provider.force_flush()
    tally = {}
    for level in verdicts:
        tally[level] = tally.get(level, 0) + 1
    judged = [level for level in verdicts if level in LEVELS]
    summary = {"experiment": args.experiment, "examples": len(dataset), "judge_levels": tally,
               "mean_usefulness": (sum(USEFULNESS_SCORE[level] for level in judged) / len(judged)) if judged else None,
               "redundant_rate": (sum(level == "redundant" for level in judged) / len(judged)) if judged else None,
               "prompt_sha256": prompt_sha}
    print(json.dumps(summary, indent=2))
    return 0


def env_file_from(argv):
    """Find --env-file before argparse runs, so the file's HEARTBEAT_JUDGE_* values can serve as option defaults."""
    for index, item in enumerate(argv):
        if item == "--env-file" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if item.startswith("--env-file="):
            return Path(item.split("=", 1)[1])
    return DEFAULT_ENV


def main():
    """Parse judge configuration and dispatch the requested local or Phoenix command."""
    load_env(env_file_from(sys.argv[1:]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="dotenv file to load before parsing options (existing variables win)")
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--judge-base-url", default=os.environ.get("HEARTBEAT_JUDGE_BASE_URL", DEFAULT_JUDGE_URL))
    parser.add_argument("--judge-model", default=os.environ.get("HEARTBEAT_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-provider", choices=("vllm", "openai"), default=os.environ.get("HEARTBEAT_JUDGE_PROVIDER", "vllm"))
    parser.add_argument("--judge-api-key-env", default=os.environ.get("HEARTBEAT_JUDGE_API_KEY_ENV", DEFAULT_JUDGE_KEY_ENV),
                        help="Environment variable holding the judge endpoint key")
    parser.add_argument("--thinking", action="store_true", help="Enable the model's thinking mode (slower; default off)")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--max-tokens", type=int, help="Completion token cap including reasoning; defaults to 4096 for OpenAI, 600 for vLLM")
    commands = parser.add_subparsers(dest="command", required=True)

    trial = commands.add_parser("try", help="Judge a few labeled cases locally and print the verdicts")
    trial.add_argument("--cases", type=Path, required=True)
    trial.add_argument("--labels", type=Path)
    trial.add_argument("--limit", type=int, default=4)
    trial.set_defaults(func=cmd_try)

    build = commands.add_parser("build-dataset", help="Upload cases with labels as a Phoenix dataset")
    build.add_argument("--cases", type=Path, required=True)
    build.add_argument("--labels", type=Path, help="The extractor's CSV with the label columns filled in")
    build.add_argument("--name", required=True)
    build.add_argument("--period", default="pre", help="pre or post the live change; stored as example metadata")
    build.add_argument("--include-unlabeled", action="store_true", help="Also upload cases without a decision label (for post-change scoring)")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_build_dataset)

    run = commands.add_parser("run", help="Run the judge as a Phoenix experiment on a dataset split")
    run.add_argument("--name", required=True, help="Dataset name")
    run.add_argument("--split", default="holdout", help="tune, holdout, silent, unsplit or all")
    run.add_argument("--experiment", required=True)
    run.add_argument("--reps", type=int, default=1)
    run.add_argument("--timeout", type=int, default=300, help="Seconds per judged example")
    run.add_argument("--dry-run", type=int, default=0, help="Judge only this many examples without recording the experiment")
    run.add_argument("--trace-project", default="openclaw-heartbeat-judge", help="Phoenix project for the judge's own LLM spans; empty disables")
    run.set_defaults(func=cmd_run)

    pull = commands.add_parser("pull-labels", help="Write the dataset's current labels and notes to a CSV (edits made in the Phoenix UI included)")
    pull.add_argument("--name", required=True, help="Dataset name")
    pull.add_argument("--version-id", help="A specific dataset version; default is the latest")
    pull.add_argument("--out", type=Path, help="CSV to write (default labels-<dataset>.csv next to --merge-into, or under runs/)")
    pull.add_argument("--merge-into", type=Path, help="The extractor's cases.csv; its label columns are updated in place by case_id")
    pull.set_defaults(func=cmd_pull_labels)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
