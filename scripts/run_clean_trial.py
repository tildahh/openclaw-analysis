"""Run one prepared Mac-mini trial through its own temporary evaluation gateway."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from clean_task_config import (BLOCKED_TOOLS, CASE_TIMEOUT, GATEWAY_PORT, GATEWAY_URL,
                               OPENCLAW_EXECUTABLE, build_trial_command,
                               build_trial_gateway_command)
from scripts.prepare_task_workspace import hash_inventory, inventory_workspace


def timestamp_now():
    """Return a UTC audit timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_private_json(path, value):
    """Write an audit artifact without exposing its contents in terminal output."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)


def validate_trial(trial):
    """Reject unprepared or production-connected configuration before launching anything."""
    if not trial.is_absolute() or trial.resolve() != trial or not (trial / "config.json").is_file():
        raise ValueError("Use an existing absolute trial directory without symlink aliases")
    for name in ("workspace", "state", "config.json", "evidence"):
        path = trial / name
        if path.is_symlink() or (name in ("workspace", "state") and not path.is_dir()):
            raise ValueError("Trial workspace, state, config and evidence must be separate local paths")
    manifest = json.loads((trial / "workspace.manifest.json").read_text())
    if manifest.get("format") != "task-workspace-trial-v1" or manifest.get("workspace") != str(trial / "workspace") or hash_inventory(inventory_workspace(trial / "workspace")) != manifest.get("seed_sha256"):
        raise ValueError("Trial workspace differs from its approved seed manifest")
    config = json.loads((trial / "config.json").read_text())
    agents = config["agents"]
    if set(agents["entries"]) != {"minilda"}:
        raise ValueError("Only the minilda evaluation agent may be configured")
    agent = agents["entries"]["minilda"]
    for settings in (agents["defaults"], agent):
        if settings.get("workspace") != str(trial / "workspace") or settings.get("heartbeat", {}).get("every") != "0m":
            raise ValueError("Workspace or heartbeat configuration does not match this trial")
    if agent.get("agentDir") != str(trial / "state/agents/minilda/agent") or config.get("session", {}).get("store") != str(trial / "state/agents/{agentId}/sessions/sessions.json"):
        raise ValueError("Agent and session state must belong to this trial")
    gateway = config.get("gateway", {})
    auth = gateway.get("auth", {})
    if not isinstance(auth, dict) or set(auth) != {"mode", "token"} or auth.get("mode") != "none" or not isinstance(auth.get("token"), str) or not re.fullmatch(r"[0-9a-f]{64}", auth["token"]):
        raise ValueError("Trial auth must use mode none with one pre-generated private 64-hex token")
    if gateway.get("mode") != "local" or gateway.get("bind") != "loopback" or gateway.get("port") != GATEWAY_PORT or gateway.get("tailscale", {}).get("mode") != "off" or gateway.get("remote"):
        raise ValueError("Gateway must use the dedicated unauthenticated loopback endpoint")
    if config.get("channels") != {} or config.get("cron", {}).get("enabled") is not False:
        raise ValueError("Trial channels and cron must be disabled")
    plugins = config.get("plugins", {}).get("entries", {})
    if plugins.get("telegram", {}).get("enabled") is not False or plugins.get("phoenix-eval-observer", {}).get("config", {}).get("outputDir") != str(trial / "evidence/observer"):
        raise ValueError("Telegram must be disabled and observer output must stay in trial evidence")
    for tools in (config.get("tools", {}), agent.get("tools", {})):
        if not set(BLOCKED_TOOLS).issubset(tools.get("deny", [])):
            raise ValueError("Required outbound-action restrictions are missing")
    evidence = trial / "evidence"
    if any((evidence / name).exists() for name in ("launch-started.json", "result.json")):
        raise ValueError("This trial has already been attempted; prepare a fresh clone")
    return hashlib.sha256((trial / "config.json").read_bytes()).hexdigest()


def build_clean_environment(overrides):
    """Remove inherited gateway routing and select only the prepared trial state/config."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("OPENCLAW_GATEWAY_")}
    environment.update(overrides)
    return environment


def assert_port_available():
    """Refuse to launch when another process already owns the evaluation port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", GATEWAY_PORT))


def assert_owned_listener(process):
    """Verify the listening gateway belongs to the process group this runner created."""
    if process.poll() is not None:
        raise RuntimeError("Evaluation gateway exited before the request")
    result = subprocess.run(["/usr/sbin/lsof", "-nP", "-t", f"-iTCP:{GATEWAY_PORT}", "-sTCP:LISTEN"],
                            text=True, capture_output=True, timeout=3)
    listeners = [int(value) for value in result.stdout.split()]
    if not listeners or any(os.getpgid(pid) != process.pid for pid in listeners):
        raise RuntimeError("Evaluation port is not owned exclusively by this trial")


def wait_gateway_ready(process, environment, evidence):
    """Check the owned listener and native health response for at most thirty seconds."""
    deadline = time.monotonic() + 30
    # An explicit --url requires a token even for auth:none; validated private config owns routing.
    command = [OPENCLAW_EXECUTABLE, "gateway", "call", "health", "--json", "--timeout", "2500"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Evaluation gateway failed to start; inspect gateway.log")
        try:
            with socket.create_connection(("127.0.0.1", GATEWAY_PORT), timeout=1):
                pass
            assert_owned_listener(process)
            health = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=5)
            if health.returncode == 0:
                payload = json.loads(health.stdout)
                if payload.get("ok") is True:
                    write_private_json(evidence / "health.json", payload)
                    return
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError("Evaluation gateway was not healthy within 30 seconds")


def stop_owned_gateway(process):
    """Terminate only the dedicated process group created by this invocation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=10)
        return "terminated"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        return "killed_after_timeout"


def interrupt_trial(_signal, _frame):
    """Turn runner termination into the same audited cleanup path as a keyboard interrupt."""
    raise KeyboardInterrupt


def classify_trial_result(response, exit_code):
    """Keep observed timeout and aborted outcomes distinct from ordinary execution failures."""
    status = response.get("status")
    metadata = response.get("result", {}).get("meta", {})
    if status == "timeout" or metadata.get("stopReason") == "timeout":
        return "timeout"
    if status == "aborted" or metadata.get("aborted") is True:
        return "aborted"
    return "completed" if exit_code == 0 and status == "ok" else "failed"


def run_trial(trial, prompt, session_key, run_id, execute=False, model=None):
    """Preview or execute one fixed-budget trial while preserving every artifact."""
    configuration_hash = validate_trial(trial)
    gateway = build_trial_gateway_command(str(trial))
    request = build_trial_command(str(trial), session_key, prompt, run_id, model)
    summary = {"status": "preview", "trial": str(trial), "session_key": session_key,
               "run_id": run_id, "gateway_url": GATEWAY_URL, "timeout_seconds": CASE_TIMEOUT,
               "configuration_sha256": configuration_hash, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    if not execute:
        return summary
    assert_port_available()
    evidence = trial / "evidence"
    evidence.mkdir(mode=0o700, exist_ok=True)
    summary.update(status="starting", started_at=timestamp_now())
    with (evidence / "launch-started.json").open("x") as marker:
        os.chmod(marker.name, 0o600)
        json.dump(summary, marker)
    environment = build_clean_environment(gateway["env"])
    started, process = time.monotonic(), None
    try:
        with (evidence / "gateway.log").open("w") as log:
            os.chmod(log.name, 0o600)
            process = subprocess.Popen(gateway["command"], env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=trial / "workspace")
        summary["gateway_pid"] = process.pid
        write_private_json(evidence / "process.json", summary)
        wait_gateway_ready(process, environment, evidence)
        assert_owned_listener(process)
        if hashlib.sha256((trial / "config.json").read_bytes()).hexdigest() != configuration_hash:
            raise ValueError("Trial configuration changed during gateway startup; task was not submitted")
        summary.update(status="running", request_started_at=timestamp_now())
        write_private_json(evidence / "process.json", summary)
        with (evidence / "response.json").open("w") as output, (evidence / "request-stderr.log").open("w") as error:
            os.chmod(output.name, 0o600)
            os.chmod(error.name, 0o600)
            result = subprocess.run(request["command"], env=environment, stdout=output, stderr=error,
                                    timeout=CASE_TIMEOUT + 30, cwd=trial / "workspace")
        response = json.loads((evidence / "response.json").read_text())
        summary.update(status=classify_trial_result(response, result.returncode),
                       exit_code=result.returncode, gateway_status=response.get("status"), gateway_run_id=response.get("runId"))
    except (Exception, KeyboardInterrupt) as error:
        outcome = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "aborted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary.update(status=outcome, error_type=type(error).__name__)
        write_private_json(evidence / "runner-error.json", {"type": type(error).__name__, "message": str(error)})
    finally:
        summary.update(finished_at=timestamp_now(), elapsed_seconds=round(time.monotonic() - started, 3))
        write_private_json(evidence / "result.json", summary)
        if process is not None:
            try:
                summary["gateway_cleanup"] = stop_owned_gateway(process)
            except Exception as error:
                summary.update(status="failed", gateway_cleanup="failed", cleanup_error_type=type(error).__name__)
        summary["cleanup_finished_at"] = timestamp_now()
        write_private_json(evidence / "result.json", summary)
    return summary


def main():
    """Parse a prepared-trial request; execution requires an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    previous_handler = signal.signal(signal.SIGTERM, interrupt_trial)
    try:
        result = run_trial(args.trial, args.prompt_file.read_text(), args.session_key, args.run_id, args.execute, args.model)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("completed", "preview") else 1


if __name__ == "__main__":
    raise SystemExit(main())
