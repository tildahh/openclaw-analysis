"""Helpers for cookbook.ipynb.

Every function here reads plain pandas frames, so the notebook runs from the saved
files in results/ without OpenClaw, the Spark or Phoenix. The optional live cells in
the notebook pass Phoenix client frames to check_tool_spans / find_root_span_ids /
build_check_annotations / build_usefulness_annotations, which return frames in the shape
that Client.spans.log_span_annotations_dataframe expects (span_id, label, score, explanation).
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

try:  # inside Jupyter these render; outside they fall back to print
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover
    Markdown = None

    def display(obj):
        """Print an object when IPython is not installed."""
        print(obj)


# --------------------------------------------------------------------------- files

# Result file -> the script or step that produces it. Used by show_result_status() and by
# the "Pending" note load_result() shows when a file is missing.
PRODUCERS = {
    "summary.json": "scripts/extract_heartbeats.py (writes runs/heartbeats-<stamp>/summary.json)",
    "failure_notes.csv": "hand-written open-coding notes, one row per trace (trace_id, last_ok, first_failed, note)",
    "labels.csv": "the labeled cases.csv from scripts/extract_heartbeats.py, after the annotation pass",
    "judge_holdout.csv": "python3 -m scripts.judge_heartbeats run --split holdout (export the experiment: human_label, judge_label)",
    "heartbeats_2026-09-28.csv": "the day's heartbeats pulled from Phoenix (runs/heartbeat-after-inventory-*/ and the baseline folder)",
    "decision_cases.csv": "python3 -m scripts.run_decision_cases run (export both experiments: case_id, variant, repetition, passed)",
    "judged_heartbeats.csv": "heartbeats_2026-09-28.csv joined with the heartbeat_usefulness_sol_v1 annotations",
    "heartbeat_wording_observations.json": "verified native observations after the wording change; delivery evidence is recorded separately",
    "task_comparison.json": "completed nine-task comparison reviewed against captured evidence, not new judge calls",
}

HEARTBEAT_STEPS = ["context built", "tools chosen", "tools succeeded", "decision made", "reply delivered or suppressed"]

PERIOD_ORDER = ["shared_whole_day", "shared_restarted", "own_session", "own_session_new_words", "isolation_and_wording", "own_session_new_words_late_hours", "isolation_wording_fresh_lookup"]
PERIOD_NAMES = {
    "shared_whole_day": "Shared conversation, whole day",
    "shared_restarted": "Shared conversation, just restarted",
    "own_session": "Own session per poll",
    "own_session_new_words": "Own session, new wording",
    "own_session_new_words_late_hours": "New wording, extended hours",
    "isolation_and_wording": "Own session, new words",
    "isolation_wording_fresh_lookup": "Own session, new words + fresh lookup",
}


def _show_markdown(text: str) -> None:
    """Render text as Markdown in Jupyter, or print it anywhere else."""
    display(Markdown(text) if Markdown else text)


def show_result_status(results_dir: Path | str) -> None:
    """Show which result files exist, so an unfinished copy of the notebook reads as a checklist."""
    results_dir = Path(results_dir)
    rows = [
        {"file": name, "status": "present" if (results_dir / name).exists() else "pending", "produced by": producer}
        for name, producer in PRODUCERS.items()
    ]
    # Displayed rather than returned: a returned frame would print a second time as the cell's value.
    display(pd.DataFrame(rows).set_index("file"))


def load_result(results_dir: Path | str, name: str):
    """Return a DataFrame (csv) or dict (json) from results/, or None with a Pending note."""
    path = Path(results_dir) / name
    if not path.exists():
        _show_markdown(f"> **Pending:** `{name}` is not in `{Path(results_dir)}/` yet. Produced by: {PRODUCERS.get(name, 'a later step')}.")
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)


# --------------------------------------------------------------------------- section 2

def build_summary_table(summary: dict) -> pd.DataFrame:
    """Flatten scripts/extract_heartbeats.py's summary.json into a two-column table."""
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub, val in value.items():
                rows.append({"measure": f"{key}.{sub}", "value": val})
        elif isinstance(value, list):
            rows.append({"measure": key, "value": ", ".join(map(str, value))})
        else:
            rows.append({"measure": key, "value": value})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- section 3

def build_transition_matrix(notes: pd.DataFrame, steps=HEARTBEAT_STEPS) -> pd.DataFrame:
    """Count runs by last step that went right (rows) and first step that went wrong (columns).

    Expects columns last_ok and first_failed holding step names from `steps`.
    A run that went right all the way has first_failed = "none".
    """
    cats = list(steps) + ["none"]
    last_ok = pd.Categorical(notes["last_ok"], categories=cats)
    first_failed = pd.Categorical(notes["first_failed"], categories=cats)
    return pd.crosstab(last_ok, first_failed, rownames=["last step ok"], colnames=["first step failed"], dropna=False)


def count_labels(labels: pd.DataFrame, column: str = "decision") -> pd.DataFrame:
    """Count each label in one column, with its share of all cases."""
    counts = labels[column].fillna("(blank)").value_counts()
    frame = counts.rename("cases").to_frame()
    frame["share"] = (frame["cases"] / frame["cases"].sum()).round(2)
    return frame


# --------------------------------------------------------------------------- section 4

USEFULNESS_SCORE = {"useful": 1.0, "somewhat_useful": 0.5, "redundant": 0.0}


def compare_judge_to_labels(holdout: pd.DataFrame, human: str = "human_label", judge: str = "judge_label"):
    """Return judge-versus-human agreement on the holdout, plus the confusion matrix.

    Exact agreement counts every label. Within-one and redundant-or-not use the three
    usefulness levels only; silent labels (correct_silence, missed, unsure) are compared
    exactly and left out of the distance numbers.
    """
    df = holdout.dropna(subset=[human, judge]).copy()
    n = len(df)
    exact = (df[human] == df[judge]).mean() if n else float("nan")
    levels = df[df[human].isin(USEFULNESS_SCORE) & df[judge].isin(USEFULNESS_SCORE)]
    dist = (levels[human].map(USEFULNESS_SCORE) - levels[judge].map(USEFULNESS_SCORE)).abs()
    within_one = (dist <= 0.5).mean() if len(levels) else float("nan")
    redundant = ((levels[human] == "redundant") == (levels[judge] == "redundant")).mean() if len(levels) else float("nan")
    baseline = df[human].value_counts(normalize=True).max() if n else float("nan")
    agreement = pd.DataFrame(
        [
            {"measure": "holdout cases", "value": n},
            {"measure": "exact agreement", "value": round(exact, 3)},
            {"measure": "within one level (usefulness levels only)", "value": round(within_one, 3)},
            {"measure": "redundant-or-not agreement", "value": round(redundant, 3)},
            {"measure": "always-guess-the-most-common-label baseline", "value": round(baseline, 3)},
        ]
    )
    confusion = pd.crosstab(df[human], df[judge], rownames=["human"], colnames=["judge"])
    return agreement, confusion


# --------------------------------------------------------------------------- section 5 and 7

def _format_range(series: pd.Series, suffix: str = "") -> str:
    """Format the numbers in a column as "lo to hi", or one number when they are all equal."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return ""
    lo, hi = int(s.min()), int(s.max())
    return f"{lo}{suffix}" if lo == hi else f"{lo} to {hi}{suffix}"


def _format_thousands(value) -> str:
    """Format a token count as whole thousands, such as 188k."""
    if pd.isna(value):
        return ""
    return f"{value / 1000:.0f}k"


def _format_latency(seconds) -> str:
    """Format seconds as minutes and whole seconds, truncated the way Phoenix displays latency."""
    if pd.isna(seconds):
        return ""
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _format_mean(value) -> str:
    """Format a mean with two decimals, rounding halves up (0.125 -> 0.13) to match the cookbook table."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def summarize_periods(heartbeats: pd.DataFrame, period: str = "period") -> pd.DataFrame:
    """Build the before/after table from the per-heartbeat file, one column per period.

    Expected columns: period, history_messages, outcome (silent | report_ending_in_no_reply |
    message), repeated_lines_pct, same_content_new_words, tool_calls, tokens_first_call,
    tokens_total, latency_s; optional judge_label.
    """
    df = heartbeats.copy()
    periods = [p for p in PERIOD_ORDER if p in set(df[period])] + [p for p in df[period].unique() if p not in PERIOD_ORDER]
    table = {}
    for p in periods:
        g = df[df[period] == p]
        sent = g[g["outcome"] != "silent"]
        new_words = int(pd.to_numeric(sent["same_content_new_words"], errors="coerce").fillna(0).sum()) if "same_content_new_words" in sent else 0
        col = {
            "Heartbeats": len(g),
            "History in the prompt (messages)": _format_range(g["history_messages"]) if g["history_messages"].notna().any() else "",
            "Bare NO_REPLY output (delivery separate)": f"{(g['outcome'] == 'silent').sum()} of {len(g)}",
            "Report ending in NO_REPLY": f"{(sent['outcome'] == 'report_ending_in_no_reply').sum()} of {len(sent)} text responses",
            "Lines repeated from the previous message": _format_range(sent["repeated_lines_pct"], "%"),
            "Same content generated again in new words": f"{new_words} of {len(sent)} text responses",
            "Tool calls per heartbeat": _format_range(g["tool_calls"]),
            "Tokens, first model call (median)": _format_thousands(pd.to_numeric(g["tokens_first_call"], errors="coerce").median()),
            "Tokens, whole heartbeat (median)": _format_thousands(pd.to_numeric(g["tokens_total"], errors="coerce").median()),
            "Latency (median)": _format_latency(pd.to_numeric(g["latency_s"], errors="coerce").median()),
        }
        if "judge_label" in g:
            scored = sent[sent["judge_label"].isin(USEFULNESS_SCORE)]
            if len(scored):
                col["Text responses the judge called redundant"] = f"{(scored['judge_label'] == 'redundant').sum()} of {len(scored)} scored"
                col["Mean judge usefulness of text responses"] = _format_mean(scored["judge_label"].map(USEFULNESS_SCORE).mean())
            else:
                col["Text responses the judge called redundant"] = "not scored"
                col["Mean judge usefulness of text responses"] = "not scored"
        table[PERIOD_NAMES.get(p, p)] = col
    return pd.DataFrame(table)


# --------------------------------------------------------------------------- section 6

def compute_pass_hat_k(runs: pd.DataFrame, case: str = "case_id", variant: str = "variant", passed: str = "passed") -> pd.DataFrame:
    """Count repetitions and passes per case and variant, and whether every repetition passed (pass^k).

    A case counts as fixed only when all repetitions pass, so wording that works some of the
    time does not count. The last row sums the pass^k cases per variant.
    """
    df = runs.copy()
    df[passed] = df[passed].astype(str).str.lower().isin(["1", "true", "yes", "pass", "passed"]) | (df[passed] == True)  # noqa: E712
    grouped = df.groupby([case, variant])[passed].agg(reps="count", passes="sum")
    grouped["pass_all"] = grouped["passes"] == grouped["reps"]
    wide = grouped.unstack(variant)
    total = pd.DataFrame({("pass_all", v): [int(grouped.xs(v, level=variant)["pass_all"].sum())] for v in df[variant].unique()}, index=["cases passing all repetitions"])
    return pd.concat([wide, total])


# --------------------------------------------------------------------------- section 7: code checks on spans

def _find_column(frame: pd.DataFrame, *candidates: str):
    """Return the first candidate column the frame has, or None."""
    for c in candidates:
        if c in frame.columns:
            return c
    return None


def check_tool_spans(spans: pd.DataFrame, trace_ids, budget: int = 10) -> pd.DataFrame:
    """Count tool calls, tool errors and repeated identical calls per heartbeat trace, and flag the budget.

    Only the traces in trace_ids are checked: the budget of 10 comes from the heartbeat
    instructions, so a conversation or task run in the same time window has no such limit.
    A heartbeat without tool spans still gets a row, with zero calls. Works on the frame
    returned by Client.spans.get_spans_dataframe; column names differ slightly between
    client versions, so the likely names are tried in order.
    """
    trace = _find_column(spans, "context.trace_id", "trace_id")
    status = _find_column(spans, "status_code")
    tool_name = _find_column(spans, "attributes.tool.name", "attributes.gen_ai.tool.name", "attributes.openclaw.toolName")
    tool_input = _find_column(spans, "attributes.gen_ai.tool.call.arguments", "attributes.openclaw.content.tool_input", "attributes.input.value")
    wanted = set(pd.Series(trace_ids).dropna().astype(str))
    heartbeat_spans = spans[spans[trace].astype(str).isin(wanted)]
    tools = heartbeat_spans[heartbeat_spans["name"] == "openclaw.tool.execution"]
    rows = []
    for tid in heartbeat_spans[trace].unique():
        g = tools[tools[trace] == tid]
        errors = int((g[status].astype(str).str.upper() == "ERROR").sum()) if status else 0
        key = g[tool_name].astype(str) + "|" + (g[tool_input].astype(str) if tool_input else "")
        rows.append({"trace_id": tid, "tool_calls": len(g), "tool_errors": errors,
                     "repeated_calls": int(key.duplicated().sum()), "over_budget": len(g) > budget})
    if not rows:
        return pd.DataFrame(columns=["tool_calls", "tool_errors", "repeated_calls", "over_budget"])
    return pd.DataFrame(rows).set_index("trace_id")


def find_root_span_ids(spans: pd.DataFrame) -> pd.Series:
    """Map each trace_id to its root span id (the span with no parent)."""
    trace = _find_column(spans, "context.trace_id", "trace_id")
    span = _find_column(spans, "context.span_id", "span_id")
    roots = spans[spans["parent_id"].isna()]
    # The client may keep span ids in the index instead of a column.
    span_ids = roots[span].to_numpy() if span else roots.index.to_numpy()
    return pd.Series(span_ids, index=roots[trace].to_numpy())


def build_check_annotations(checks: pd.DataFrame, roots: pd.Series, column: str, pass_label: str, fail_label: str, score_column: str | None = None) -> pd.DataFrame:
    """Build one CODE annotation per checked trace, on its root span, from a boolean or count column."""
    rows = []
    for tid, row in checks.iterrows():
        if tid not in roots.index:
            continue
        failed = float(row[column]) > 0  # True/False and counts both work
        rows.append({
            "span_id": roots[tid],
            "label": fail_label if failed else pass_label,
            "score": float(row[score_column]) if score_column else (0.0 if failed else 1.0),
            "explanation": f"{column}={row[column]}; tool_calls={row.get('tool_calls', '')}",
        })
    return pd.DataFrame(rows, columns=["span_id", "label", "score", "explanation"])


# --------------------------------------------------------------------------- section 8

def build_usefulness_annotations(judged: pd.DataFrame) -> pd.DataFrame:
    """Build LLM annotations (span_id, label, score, explanation) from judged_heartbeats.csv.

    The score and explanation columns are optional: without judge_score the label's usefulness
    level is used, and without judge_explanation the annotation carries no explanation.
    """
    span = _find_column(judged, "span_id", "root_span_id")
    df = judged[judged["judge_label"].notna() & (judged["judge_label"].astype(str) != "")]
    score = pd.to_numeric(df["judge_score"], errors="coerce") if "judge_score" in df else df["judge_label"].map(USEFULNESS_SCORE)
    explanation = df["judge_explanation"] if "judge_explanation" in df else ""
    return pd.DataFrame({
        "span_id": df[span],
        "label": df["judge_label"],
        "score": score,
        "explanation": explanation,
    }).reset_index(drop=True)
