"""Plot completion counts from the scatter plot's saved, matched attempts.

Run: python scripts/figures/task_completion_grid.py
Reads results/model-completion-time.json; makes no model or judge calls.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from palette import BG, INK, MODELS, MUTED, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/model-completion-time.json"
FIGURE = ROOT / "figures/task-completion-grid"
FILLS = ["#EEF0F2", "#BDD7EA", "#66A6D1", "#226FA6"]
LABELS = {  # two-line column headings for the models in palette.MODELS
    "Qwen-3.6 35B": "Qwen-3.6\n35B",
    "Qwen-3.5 122B": "Qwen-3.5\n122B",
    "Nemotron 120B": "Nemotron\n120B",
    "Muse 30B": "Muse\n30B",
    "GLM 5.3 Flash 320B": "GLM 5.3\nFlash 320B",
    "DeepSeek V4 Flash 284B": "DeepSeek V4\nFlash 284B",
}
TASKS = {
    "stockholm-holiday-flight-shortlist": "Flight shortlist",
    "weekly-course-readings": "Weekly course readings",
    "academic-week-study-plan": "Study plan",
    "cal-hacks-follow-up": "Event follow-up",
    "coreweave-conference-preparation": "Conference prep",
    "benchmark-discovery-brief": "Find an agent benchmark",
    "vals-hackathon-preparation": "Hackathon prep",
    "completed-coursework-followup": "Assignment preparation",
    "lora-trainable-fraction": "LoRA parameter calculation",
}
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}


def build_grid(data):
    """Count binary completion scores and reconcile every cell with the saved scatter data."""
    attempts = data["attempts"]
    assert len(attempts) == 162, "Expected six models, nine tasks and three attempts"
    assert len({(a["source_experiment"], a["source_run_id"]) for a in attempts}) == 162, "Duplicate attempt"
    groups = defaultdict(list)
    for attempt in attempts:
        assert attempt["task_completion"] in (0, 1), "Completion score is not binary"
        expected_label = "complete" if attempt["task_completion"] == 1 else "incomplete"
        assert attempt["score_label"] == expected_label or (
            attempt["score_label"] == "no_answer" and attempt["error"] and attempt["task_completion"] == 0
        ), "Completion score and label disagree"
        groups[attempt["model"], attempt["case"]].append(attempt)
    assert set(groups) == {(model, task) for model in MODELS for task in TASKS}, "Unexpected model or task"
    summary = {row["model"]: row for row in data["models"]}
    assert len(data["models"]) == 6 and set(summary) == set(MODELS), "Model summaries differ"
    counts = []
    for task in TASKS:
        row = []
        for model in MODELS:
            selected = groups[model, task]
            assert len(selected) == 3 and {a["pass"] for a in selected} == {1, 2, 3}, "Missing or duplicate pass"
            count = sum(a["task_completion"] == 1 for a in selected)
            assert math.isclose(count / 3, summary[model]["task_means"][task]), "Task mean differs from scatter data"
            row.append(count)
        counts.append(row)
    for column, model in enumerate(MODELS):
        assert math.isclose(sum(row[column] for row in counts) / 27,
                            summary[model]["mean_task_completion"]), "Model mean differs from scatter data"
    return counts


def draw_grid(counts):
    """Draw the grid with explicit counts, hardest task first, best model left, totals on both margins, and a colour tab per model column."""
    fig, ax = plt.subplots(figsize=(WIDTH, 6.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    order = sorted(range(len(counts)), key=lambda i: (sum(counts[i]), i))  # hardest task at the top
    col_order = sorted(range(6), key=lambda c: (-sum(row[c] for row in counts), c))  # best model at the left
    counts = [[counts[i][c] for c in col_order] for i in order]
    task_names = [list(TASKS.values())[i] for i in order]
    model_names = [list(MODELS)[c] for c in col_order]
    ax.imshow(counts, cmap=ListedColormap(FILLS), norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4),
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(6), [LABELS[model] for model in model_names], fontsize=11, fontweight="bold")
    ax.set_yticks(range(9), task_names, fontsize=11.5, color=INK)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=12)
    for tick in ax.get_xticklabels():
        tick.set_color(INK)
        tick.set_linespacing(1.4)
    for x, model in enumerate(model_names):  # a colour tab above each column carries the model identity instead of coloured text
        ax.add_patch(Rectangle((x - .38, -.5 - .17), .76, .07, color=MODELS[model], clip_on=False, zorder=5))
    for y, row in enumerate(counts):  # totals per task and per model, so the grid reads at a glance
        ax.text(5.68, y, f"{sum(row)}/18", ha="left", va="center", fontsize=11, color=MUTED, clip_on=False)
    for x in range(6):
        ax.text(x, 8.78, f"{sum(row[x] for row in counts)}/27", ha="center", va="center", fontsize=11, color=MUTED, clip_on=False)
    ax.text(5.68, -0.5 - .22, "per task", ha="left", va="bottom", fontsize=10, color=MUTED, clip_on=False)
    ax.text(-0.6, 8.78, "per model", ha="right", va="center", fontsize=10, color=MUTED, clip_on=False)
    ax.set_xticks([x - .5 for x in range(7)], minor=True)
    ax.set_yticks([y - .5 for y in range(10)], minor=True)
    ax.grid(which="minor", color=BG, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, row in enumerate(counts):
        for x, count in enumerate(row):
            ax.text(x, y, f"{count}/3", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if count == 3 else INK)
    place_title(fig, "Task completion by task and model")
    fig.subplots_adjust(left=.23, right=.925, top=plot_top(fig, legend=True), bottom=plot_bottom(fig, xlabel=False))
    for extension in ("png", "svg"):
        fig.savefig(FIGURE.with_suffix("." + extension), facecolor=BG)
    plt.close(fig)


def main():
    """Verify the saved sources, render the grid, and print its counts for review."""
    data = json.loads(SOURCE.read_text())
    for name, digest in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Saved source changed: {name}"
    counts = build_grid(data)
    with plt.rc_context(FONTS):
        draw_grid(counts)
    print("Task | " + " | ".join(MODELS))
    for task, row in zip(TASKS.values(), counts):
        print(task + " | " + " | ".join(f"{count}/3" for count in row))
    print("Verified all 162 attempts and all six scatter means; wrote", FIGURE.with_suffix(".png"), "and .svg")


if __name__ == "__main__":
    main()
