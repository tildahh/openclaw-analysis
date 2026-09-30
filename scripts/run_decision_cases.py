"""Controlled experiment on the offline decision fixtures: does a heartbeat policy change what the model decides to send?

The six cases in messaging-cases.json each fix a trigger, the facts known at decision time, and the
expected action (reply, notify or suppress) with checks on the message. The actor is the model alone,
no tools, no memory: what is tested is the instruction, not the whole assistant. Two arms share
everything except the policy text appended to the actor instructions.

    python3 -m scripts.run_decision_cases try --arm candidate --policy-file policies/heartbeat-silence.md --limit 2
    python3 -m scripts.run_decision_cases build-dataset --name openclaw-decision-cases-v1
    python3 -m scripts.run_decision_cases run --name openclaw-decision-cases-v1 --arm baseline --experiment decision-baseline --reps 3
    python3 -m scripts.run_decision_cases run --name openclaw-decision-cases-v1 --arm candidate --policy-file policies/heartbeat-silence.md \
        --experiment decision-candidate --reps 3

Evaluators: action_match (the decision), message_when_needed (silence when suppress, text otherwise),
no_sentinel_in_message, no_greeting_to_poll, and checks_met, an LLM check of the case's own grader
checks against the message (skipped with --no-llm-evals). Credentials come from ./.env.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from scripts.extract_heartbeats import GREETING, SENTINEL_LINE

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "datasets" / "messaging-cases.json"
DEFAULT_ENV = ROOT / ".env"
ACTIONS = ("reply", "notify", "suppress")
DEFAULT_ACTOR_URL, DEFAULT_ACTOR_MODEL, DEFAULT_ACTOR_KEY_ENV = "http://dgx-spark:8002/v1", "qwen3.5-122b-a10b", "SPARK_API_KEY"
CHECK_PROMPT = """You are verifying a message an assistant proposed to send to its user.

Message:
<message>
{message}
</message>

For each check below, answer whether the message satisfies it. Judge only the text of the message.
Checks:
{checks}

Answer with one JSON object and nothing else: {{"results": [{{"check": "<check text>", "passed": true | false, "why": "<short reason>"}}]}}"""


# ----- configuration and model access ------------------------------------------------------------

def load_env(path):
    """KEY=VALUE lines; variables already set win; comments and blanks are skipped."""
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


def make_caller(base_url, model, api_key, thinking=False, temperature=0.0, max_tokens=800):
    """One OpenAI-compatible chat call: system + user messages in, text out."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    extra = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}

    def call(system, user):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, extra_body=extra)
        return response.choices[0].message.content or ""
    return call


def first_json_object(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty reply")
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object in reply")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:].strip())
    if not isinstance(parsed, dict):
        raise ValueError("Reply is not a JSON object")
    return parsed


# ----- cases and arms -----------------------------------------------------------------------------

def load_cases(path=CASES_PATH):
    data = json.loads(Path(path).read_text())
    return data, [case for case in data["cases"]]


def actor_system_prompt(instructions, policy_text=None):
    """Baseline is the fixture's own actor instructions; the candidate appends the policy under test and nothing else."""
    return instructions if not policy_text else instructions.rstrip() + "\n\nHeartbeat policy in force:\n" + policy_text.strip()


def actor_user_prompt(actor):
    """The fixture as the model sees it: facts and request only; grader references are never included."""
    return "Situation (JSON):\n" + json.dumps(actor, indent=2, ensure_ascii=False) + "\n\nDecide and answer with one JSON object: {\"action\": \"reply\" | \"notify\" | \"suppress\", \"message\": \"<text to send>\" | null}"


def parse_decision(text):
    parsed = first_json_object(text)
    action = str(parsed.get("action", "")).strip().lower()
    if action not in ACTIONS:
        raise ValueError("action must be one of " + ", ".join(ACTIONS) + "; got " + repr(parsed.get("action")))
    message = parsed.get("message")
    message = None if message in (None, "", "null") else str(message)
    return {"action": action, "message": message}


def decide(case_input, call, policy_text=None):
    """Run the actor once; keep the raw reply for debugging."""
    raw = call(actor_system_prompt(case_input["actor_instructions"], policy_text), actor_user_prompt(case_input["actor"]))
    try:
        return {**parse_decision(raw), "raw": raw[:2000]}
    except ValueError as error:
        return {"action": "unparsed", "message": None, "error": str(error), "raw": raw[:2000]}


def example_from_case(case, instructions):
    """Phoenix example: the actor's view as input, the grader reference (minus source events) as expected."""
    reference = {key: value for key, value in case["grader_reference"].items() if key != "source_events"}
    return {"input": {"case_id": case["id"], "trigger": case["actor"].get("trigger"), "actor": case["actor"], "actor_instructions": instructions},
            "output": reference, "metadata": {"status": case.get("status"), "evidence_kind": reference.get("evidence_kind")}, "splits": "decision"}


# ----- evaluators (input, output, expected bound by name) ----------------------------------------

def action_match(input, output, expected):
    want, got = (expected or {}).get("expected_action"), (output or {}).get("action")
    if want not in ACTIONS:
        return {"label": "unlabeled", "explanation": "No expected action on this example."}
    return {"score": float(got == want), "label": "match" if got == want else str(got) + " vs " + want, "explanation": "actor=" + str(got) + ", expected=" + want + "."}


def message_when_needed(input, output, expected):
    """Suppress means no user-facing text; reply and notify mean some."""
    want, message = (expected or {}).get("expected_action"), (output or {}).get("message")
    if want not in ACTIONS:
        return {"label": "unlabeled", "explanation": "No expected action."}
    has_text = bool(message and message.strip())
    ok = (not has_text) if want == "suppress" else has_text
    return {"score": float(ok), "label": "ok" if ok else ("text despite suppress" if want == "suppress" else "no text"),
            "explanation": "expected " + want + "; message " + ("present" if has_text else "absent") + "."}


def no_sentinel_in_message(input, output, expected):
    message = (output or {}).get("message") or ""
    if not message.strip():
        return {"score": 1.0, "label": "no message", "explanation": "Nothing to send."}
    leak = bool(SENTINEL_LINE.search(message)) or "NO_REPLY" in message or "HEARTBEAT_OK" in message
    return {"score": 0.0 if leak else 1.0, "label": "leak" if leak else "clean", "explanation": "A control token appears in the user-facing text." if leak else "Clean."}


def no_greeting_to_poll(input, output, expected):
    message = (output or {}).get("message") or ""
    trigger = str(((input or {}).get("actor") or {}).get("trigger") or (input or {}).get("trigger") or "")
    if "heartbeat" not in trigger.lower() or not message.strip():
        return {"label": "not applicable", "explanation": "Not a heartbeat poll, or no message."}
    greets = bool(GREETING.match(message)) or bool(re.search(r"\b(what would you like|which of these|what should we tackle)\b", message, re.IGNORECASE))
    return {"score": 0.0 if greets else 1.0, "label": "poll treated as human" if greets else "ok",
            "explanation": "Greets or re-asks the user in reply to a timer." if greets else "No greeting or re-asked menu."}


def make_checks_evaluator(call):
    """LLM check of the case's own grader checks against the message; fraction passed."""

    def checks_met(input, output, expected):
        checks = [check for check in (expected or {}).get("checks", []) if not check.lower().startswith("message is null")]
        message = (output or {}).get("message") or ""
        if not checks:
            return {"label": "no checks", "explanation": "The case has no text checks."}
        if not message.strip():
            return {"score": 0.0 if (expected or {}).get("expected_action") != "suppress" else 1.0,
                    "label": "no message", "explanation": "No text to check."}
        prompt = CHECK_PROMPT.replace("{message}", message[:6000]).replace("{checks}", "\n".join("- " + check for check in checks))
        try:
            results = first_json_object(call(None, prompt)).get("results", [])
            passed = sum(1 for item in results if item.get("passed") is True)
            return {"score": passed / len(checks), "label": str(passed) + "/" + str(len(checks)),
                    "explanation": "; ".join(str(item.get("check", ""))[:60] + ": " + ("pass" if item.get("passed") else "fail") for item in results)[:500]}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return {"label": "check error", "explanation": str(error)[:200]}
    return checks_met


def make_usefulness_evaluator(judge_call, template, judge_module):
    """The calibrated heartbeat judge as an evaluator on the message the actor produced (its level as a 1 / 0.5 / 0 score)."""
    levels = tuple(getattr(judge_module, "LEVELS", ("useful", "somewhat_useful", "redundant")))
    scores = getattr(judge_module, "USEFULNESS_SCORE", {"useful": 1.0, "somewhat_useful": 0.5, "redundant": 0.0})

    def judge_usefulness(input, output, expected):
        message = (output or {}).get("message") or ""
        if not message.strip():
            return {"label": "no message", "explanation": "Nothing was sent; the usefulness judge rates sent text only."}
        actor = (input or {}).get("actor") or {}
        user_message = actor.get("user_message")
        case_input = {"case_id": (input or {}).get("case_id"), "trigger": actor.get("trigger"), "output": message, "prev_output": "",
                      "minutes_since_prev": 30, "user_activity_since_prev": bool(user_message),
                      "user_messages_between": ("[user] " + str(user_message)) if user_message else "", "runtime_silent": False}
        try:
            verdict = judge_module.judge_case(case_input, judge_call, template)
        except Exception as error:  # noqa: BLE001  (the judge module is under active change; never fail the experiment)
            return {"label": "judge error", "explanation": type(error).__name__ + ": " + str(error)[:200]}
        level = verdict.get("usefulness")
        if level not in levels:
            return {"label": str(level or "none"), "explanation": str(verdict.get("explanation", ""))[:300]}
        return {"score": scores[level], "label": level, "explanation": str(verdict.get("new_information", ""))[:300]}
    return judge_usefulness


# ----- commands -----------------------------------------------------------------------------------

def actor_from_args(args):
    return make_caller(args.actor_base_url, args.actor_model, os.environ.get(args.actor_api_key_env) if args.actor_api_key_env else None,
                       thinking=args.thinking, max_tokens=args.max_tokens)


def policy_from_args(args):
    if args.arm == "baseline":
        if args.policy_file:
            raise SystemExit("--policy-file is only valid with --arm candidate")
        return None, None
    if not args.policy_file:
        raise SystemExit("--arm candidate requires --policy-file")
    text = Path(args.policy_file).read_text()
    if not text.strip():
        raise SystemExit("Policy file is empty")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


def cmd_try(args):
    policy, policy_sha = policy_from_args(args)
    data, cases = load_cases(args.cases)
    call = actor_from_args(args)
    matches = []
    for case in cases[:args.limit]:
        example = example_from_case(case, data["actor_instructions"])
        decision = decide(example["input"], call, policy)
        want = example["output"].get("expected_action")
        matches.append(decision["action"] == want)
        print(json.dumps({"case_id": case["id"], "expected": want, "action": decision["action"], "message": decision.get("message"), "error": decision.get("error")}, indent=2, ensure_ascii=False))
    print(json.dumps({"arm": args.arm, "policy_sha256": policy_sha, "tried": len(matches), "action_matches": sum(matches)}))
    return 0


def cmd_build_dataset(args):
    data, cases = load_cases(args.cases)
    examples = [example_from_case(case, data["actor_instructions"]) for case in cases]
    if args.dry_run:
        print(json.dumps({"would_upload": len(examples), "cases": [example["input"]["case_id"] for example in examples]}, indent=2))
        return 0
    from phoenix.client import Client

    dataset = Client().datasets.create_dataset(name=args.name, dataset_description="Offline heartbeat decision fixtures (messaging-cases.json): trigger, facts, expected action and checks; the actor is the model alone.",
                                               examples=examples)
    print(json.dumps({"dataset": args.name, "uploaded": len(dataset)}, indent=2))
    return 0


def cmd_run(args):
    from phoenix.client import Client
    from phoenix.client.experiments import run_experiment

    policy, policy_sha = policy_from_args(args)
    provider = None
    if args.trace_project:
        from phoenix.otel import register
        provider = register(project_name=args.trace_project, auto_instrument=True, batch=True, verbose=False)
    call = actor_from_args(args)
    evaluators = [action_match, message_when_needed, no_sentinel_in_message, no_greeting_to_poll]
    if not args.no_llm_evals:
        evaluators.append(make_checks_evaluator(call))
    if args.with_usefulness_judge:
        from scripts import judge_heartbeats as judge_module

        template, _ = judge_module.read_prompt()
        judge_call = judge_module.make_judge(args.judge_base_url, args.judge_model, os.environ.get(args.judge_api_key_env) if args.judge_api_key_env else None,
                                             provider=args.judge_provider)
        evaluators.append(make_usefulness_evaluator(judge_call, template, judge_module))
    decisions = []

    def task(input):
        decision = decide(input, call, policy)
        decisions.append(decision["action"])
        return decision

    dataset = Client().datasets.get_dataset(dataset=args.name)
    run_experiment(dataset=dataset, task=task, evaluators=evaluators, experiment_name=args.experiment,
                   experiment_description="decision fixtures, arm=" + args.arm + (", policy " + policy_sha if policy_sha else ""),
                   experiment_metadata={"arm": args.arm, "policy_file": str(args.policy_file) if args.policy_file else None, "policy_sha256": policy_sha,
                                        "actor_model": args.actor_model, "actor_base_url": args.actor_base_url, "thinking": args.thinking},
                   repetitions=args.reps, timeout=args.timeout, dry_run=args.dry_run or False)
    if provider:
        provider.force_flush()
    tally = {}
    for action in decisions:
        tally[action] = tally.get(action, 0) + 1
    print(json.dumps({"experiment": args.experiment, "arm": args.arm, "examples": len(dataset), "actions": tally, "policy_sha256": policy_sha}, indent=2))
    return 0


def env_file_from(argv):
    for index, item in enumerate(argv):
        if item == "--env-file" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if item.startswith("--env-file="):
            return Path(item.split("=", 1)[1])
    return DEFAULT_ENV


def main():
    load_env(env_file_from(sys.argv[1:]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--actor-base-url", default=os.environ.get("HEARTBEAT_ACTOR_BASE_URL", DEFAULT_ACTOR_URL))
    parser.add_argument("--actor-model", default=os.environ.get("HEARTBEAT_ACTOR_MODEL", DEFAULT_ACTOR_MODEL))
    parser.add_argument("--actor-api-key-env", default=os.environ.get("HEARTBEAT_ACTOR_API_KEY_ENV", DEFAULT_ACTOR_KEY_ENV))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=800)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, func in (("try", cmd_try), ("run", cmd_run)):
        sub = commands.add_parser(name)
        sub.add_argument("--arm", choices=("baseline", "candidate"), required=True)
        sub.add_argument("--policy-file", type=Path)
        if name == "try":
            sub.add_argument("--limit", type=int, default=6)
        else:
            sub.add_argument("--name", required=True)
            sub.add_argument("--experiment", required=True)
            sub.add_argument("--reps", type=int, default=3)
            sub.add_argument("--timeout", type=int, default=300)
            sub.add_argument("--dry-run", type=int, default=0)
            sub.add_argument("--no-llm-evals", action="store_true")
            sub.add_argument("--with-usefulness-judge", action="store_true", help="Also score each sent message with the calibrated heartbeat judge")
            sub.add_argument("--judge-provider", default=os.environ.get("HEARTBEAT_JUDGE_PROVIDER", "openai"), choices=("vllm", "openai"))
            sub.add_argument("--judge-base-url", default=os.environ.get("HEARTBEAT_JUDGE_BASE_URL", "https://api.openai.com/v1"))
            sub.add_argument("--judge-model", default=os.environ.get("HEARTBEAT_JUDGE_MODEL", "gpt-6-sol"))
            sub.add_argument("--judge-api-key-env", default=os.environ.get("HEARTBEAT_JUDGE_API_KEY_ENV", "OPENAI_API_KEY"))
            sub.add_argument("--trace-project", default="openclaw-decision-cases")
        sub.set_defaults(func=func)

    build = commands.add_parser("build-dataset")
    build.add_argument("--name", required=True)
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_build_dataset)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
