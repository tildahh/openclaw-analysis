"""Plot completion versus response time from saved attempts selected by the model dashboard.

Run with a Python environment containing matplotlib:
    python scripts/figures/model_completion_time.py

No network access, new trials, or judge calls are used. The matched attempts and source hashes
are saved in results/model-completion-time.json alongside the PNG and SVG figures.
"""

import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
from palette import BG, GRID, INK, MODELS, MUTED, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top
from publish_setup_view import MAKEUPS, SETUPS

PUBLICATION = ROOT / "runs/model-comparison-20min-matched/publication"
JUDGES = ROOT / "results/task-llm-evals.jsonl"
COUNTS = ROOT / "results/task-judge-agreement.json"
OUT = ROOT / "results/model-completion-time.json"
FIGURE = ROOT / "figures/model-completion-time"
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}


def read_json(path, sources):
    """Read one saved source and retain its checksum for the plot's provenance record."""
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def read_attempts():
    """Match saved timing and completion scores for the dashboard's exact 162 attempt slots."""
    sources = {}
    mapping = read_json(PUBLICATION / "dataset-mapping.json", sources)
    cases = {value["dataset_example_id"]: case for case, value in mapping.items()}
    paths = {}
    for receipt in sorted(PUBLICATION.glob("*/publish-receipt.json")):
        saved = read_json(receipt, sources)
        number = int(base64.b64decode(saved["experiment_id"]).decode().split(":")[1])
        assert number not in paths, f"Duplicate experiment receipt: {number}"
        paths[number] = receipt.parent / "final-readback.json"
    paths[41] = ROOT / "runs/task-qwen35-lora-makeup/publication/phoenix-final-readback.json"
    raw = JUDGES.read_bytes()
    sources[str(JUDGES.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    judges = {}
    for row in map(json.loads, raw.decode().splitlines()):
        key = (row["experiment"], row["run_id"])
        assert key not in judges, f"Duplicate judge record: {key}"
        judges[key] = row
    expected = {row["model"]: row["judges"]["task_completion"]
                for row in read_json(COUNTS, sources)["by_model"]}
    attempts, models, replacements = [], [], []
    for model, description, experiments in SETUPS:
        rows = []
        for pass_number, experiment in enumerate(experiments, 1):
            makeups = MAKEUPS.get(experiment, {})
            originals = read_json(paths[experiment], sources)["runs"]
            selected = [(experiment, run) for run in originals
                        if cases[run["dataset_example_id"]] not in makeups]
            for run in originals:
                case = cases[run["dataset_example_id"]]
                if case in makeups:
                    replacements.append({"model": model, "pass": pass_number, "case": case,
                                         "excluded_experiment": experiment, "excluded_run_id": run["id"],
                                         "recorded_error": run["error"], "replacement_experiment": makeups[case]})
            for replacement in sorted(set(makeups.values())):
                selected.extend((replacement, run)
                                for run in read_json(paths[replacement], sources)["runs"])
            assert len(selected) == 9, (model, pass_number, "Expected nine attempts")
            for source_experiment, run in selected:
                case = cases[run["dataset_example_id"]]
                notes = {note["name"]: note for note in run["annotations"]}
                seconds = notes.get("native_seconds", {}).get("score")
                assert seconds is not None and seconds > 0, (model, case, "Missing native timing")
                elapsed = (datetime.fromisoformat(run["end_time"]) -
                           datetime.fromisoformat(run["start_time"])).total_seconds()
                assert abs(seconds - elapsed) < 0.01, (model, case, "Conflicting timing records")
                judge = judges.get((source_experiment, run["id"]))
                if judge is None:
                    # Match the publisher's zero fill only for an explicitly recorded no-answer failure.
                    assert run["error"] and not run["output"], (model, case, "Unexplained missing judge")
                    assert "task_completion" not in notes, (model, case, "Unmatched saved judge")
                    score, label, revision = 0.0, "no_answer", None
                    score_source = "dashboard zero-fill rule for recorded no-answer failure"
                else:
                    assert (judge["case"], judge["example_id"]) == (case, run["dataset_example_id"])
                    if run.get("trace_id") and judge.get("trace_id"):
                        assert run["trace_id"] == judge["trace_id"], (model, case, "Trace mismatch")
                    verdict = judge["scores"]["task_completion"]
                    score, label = verdict["score"], verdict["label"]
                    assert score is not None and 0 <= score <= 1, (model, case, "Missing/invalid completion score")
                    revision = judge.get("revisions", {}).get("task_completion", judge.get("revision", "v1"))
                    score_source = str(JUDGES.relative_to(ROOT))
                    if "task_completion" in notes:
                        assert notes["task_completion"]["score"] == score, (model, case, "Conflicting saved scores")
                rows.append({"model": model, "pass": pass_number, "case": case,
                             "source_experiment": source_experiment, "source_run_id": run["id"],
                             "source_readback": str(paths[source_experiment].relative_to(ROOT)),
                             "trace_id": (judge or {}).get("trace_id") or run.get("trace_id"),
                             "native_seconds": seconds, "task_completion": score,
                             "score_label": label, "score_source": score_source, "judge_input_revision": revision,
                             "error": run["error"], "execution_label": notes.get("execution_completed", {}).get("label"),
                             "reference_label": notes.get("answer_correctness", {}).get("label")})
        assert len(rows) == 27 and len({(row["case"], row["pass"]) for row in rows}) == 27
        assert Counter(row["case"] for row in rows) == Counter({case: 3 for case in cases.values()})
        assert sum(row["task_completion"] for row in rows) == expected[model], (model, "Dashboard total mismatch")
        task_means = {case: statistics.mean(row["task_completion"] for row in rows if row["case"] == case)
                      for case in cases.values()}
        models.append({"model": model, "source_experiments": experiments, "serving_description": description,
                       "attempts": len(rows), "median_minutes": statistics.median(row["native_seconds"] for row in rows) / 60,
                       "mean_task_completion": statistics.mean(task_means.values()), "task_means": task_means,
                       "zero_filled_attempts": sum(row["score_label"] == "no_answer" for row in rows)})
        attempts.extend(rows)
    for path in (Path(__file__), ROOT / "evals/publish_setup_view.py"):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    revisions = Counter(row["judge_input_revision"] for row in attempts if row["judge_input_revision"] is not None)
    return {"population": "Dashboard selections: six models, nine tasks, three passes, including the Qwen replacement",
            "x_aggregation": "Median native_seconds across all 27 selected attempts per model, divided by 60; setup excluded",
            "y_aggregation": "Mean task_completion over three passes within each task, then mean over the nine tasks",
            "failure_handling": "Retain measured failure durations; apply the dashboard zero for the explicit no-answer timeout",
            "issues": ["One DeepSeek timeout has no judge record; its completion score is zero under the dashboard rule.",
                       "Saved completion verdicts use mixed judge-input revision tags; no scores were rejudged.",
                       "The dashboard replaces one Qwen setup failure with experiment 41; the original slot is recorded below."],
            "judge_input_revisions": dict(revisions), "replacements": replacements,
            "models": models, "attempts": attempts, "source_sha256": sources}


def draw_plot(models):
    """Draw six labeled points and check that their labels do not overlap or leave the axes."""
    offsets = {"Qwen-3.6 35B": (-12, -23, "right"), "Qwen-3.5 122B": (12, 18, "left"),
               "Nemotron 120B": (14, -22, "left"), "Muse 30B": (0, -25, "center"),
               "GLM 5.3 Flash 320B": (-12, 22, "right"), "DeepSeek V4 Flash 284B": (14, 8, "left")}
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0.5, 1)
    ax.set_xticks(range(8))
    ax.set_yticks([i / 10 for i in range(5, 11)])
    ax.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.45, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    place_xlabel(fig, "Median time per task (minutes)")
    ax.set_ylabel("Share of attempts marked complete", fontsize=12, color=INK, labelpad=14)
    labels = []
    for row in models:
        x, y = row["median_minutes"], row["mean_task_completion"]
        color = MODELS[row["model"]]
        ax.scatter(x, y, s=260, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.scatter(x, y, s=110, color=color, edgecolor=BG, linewidth=1.8, zorder=3)
        dx, dy, align = offsets[row["model"]]
        name = TextArea(row["model"], textprops={"fontsize": 11, "fontweight": "bold", "color": color})
        label = AnnotationBbox(name, (x, y), xybox=(dx, dy), boxcoords="offset points",
                               box_alignment=({"left": 0, "center": 0.5, "right": 1}[align], 0.5),
                               frameon=False, pad=0,
                               arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.7, "lw": 0.8}, zorder=4)
        ax.add_artist(label)
        labels.append(label)
    ax.annotate("faster and more complete", xy=(0.16, 0.993), xytext=(0.42, 0.968), fontsize=10.5, color=MUTED,
                ha="left", va="center", arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.1})
    chosen = next(row for row in models if row["model"] == "GLM 5.3 Flash 320B")
    ax.scatter(chosen["median_minutes"], chosen["mean_task_completion"], s=420, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)
    ax.annotate("chosen", (chosen["median_minutes"], chosen["mean_task_completion"]), xytext=(11, -14), textcoords="offset points",
                fontsize=9.5, color=MUTED, fontstyle="italic")
    place_title(fig, "Six local models: task completion vs. time per task")
    fig.subplots_adjust(left=0.1, right=RIGHT, top=plot_top(fig), bottom=plot_bottom(fig))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.offsetbox.get_window_extent(renderer) for label in labels]
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), "Label outside plot"
        assert all(not box.overlaps(other) for other in boxes[i + 1:]), "Overlapping model labels"
    for extension in ("png", "svg"):
        fig.savefig(FIGURE.with_suffix("." + extension), facecolor=BG)
    plt.close(fig)


def main():
    """Audit the saved population, write the matched data, and render the model comparison."""
    data = read_attempts()
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    with plt.rc_context(FONTS):
        draw_plot(data["models"])
    for row in data["models"]:
        print(f"{row['model']}: n={row['attempts']}, median={row['median_minutes']:.4f} min, completion={row['mean_task_completion']:.6f}")
    print(f"Wrote {OUT.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}.png/.svg")


if __name__ == "__main__":
    main()
