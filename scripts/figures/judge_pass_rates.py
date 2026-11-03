"""Plot how often each LLM judge passed each model's answers, from the same 162 attempts as the scatter plot.

Run from the repository root:
    python3 scripts/figures/judge_pass_rates.py

Task completion and tool selection count all 27 attempts per model, and a missing answer counts as not passed,
as on the Phoenix dashboard. Faithfulness, tool response handling and retrieval relevance only score answers
that had tool results. Every pass count is checked against results/task-judge-agreement.json.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_completion_time as attempts_source
from palette import BG, GRID, INK, MODELS, MUTED, NEUTRAL_LIGHT, TEXT, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
JUDGES = ROOT / "results/task-llm-evals.jsonl"
COUNTS = ROOT / "results/task-judge-agreement.json"
FIGURE = ROOT / "figures/judge-pass-rates"
ROWS = [  # top to bottom, in descending order of each judge's median pass rate
    ("retrieval_relevance", "Retrieval relevance"),
    ("task_completion", "Task completion"),
    ("tool_selection", "Tool selection"),
    ("faithfulness", "Faithfulness"),
    ("tool_response_handling", "Tool response handling"),
]
EVERY_ATTEMPT = {"task_completion", "tool_selection"}  # scored on all 27 attempts; the other judges need tool results
TIE_WIDTH = 2.2  # pass rates closer than this, in percentage points, would draw overlapping dots
STACK_STEP = 0.19  # vertical distance between stacked dots, in row units


def count_passes():
    """Return {model: {judge: (passed, scored)}} for the scatter plot's 162 attempts, checked against the saved counts."""
    attempts = attempts_source.read_attempts()["attempts"]
    records = {}
    for line in JUDGES.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[(record["experiment"], record["run_id"])] = record
    rates = {}
    for model in MODELS:
        rows = [attempt for attempt in attempts if attempt["model"] == model]
        assert len(rows) == 27, (model, len(rows))
        rates[model] = {}
        for judge, _ in ROWS:
            passed = scored = 0
            for attempt in rows:
                record = records.get((attempt["source_experiment"], attempt["source_run_id"]), {})
                verdict = record.get("scores", {}).get(judge)
                if judge in EVERY_ATTEMPT:
                    scored += 1  # a missing answer counts as not passed, as on the dashboard
                    passed += bool(verdict and verdict["score"] == 1.0)
                elif verdict and verdict["label"] != "not_applicable":
                    scored += 1
                    passed += verdict["score"] == 1.0
            rates[model][judge] = (passed, scored)
    saved = {row["model"]: row["judges"] for row in json.loads(COUNTS.read_text())["by_model"]}
    for model, judges in rates.items():
        for judge, (passed, _) in judges.items():
            assert passed == saved[model][judge], (model, judge, passed, saved[model][judge])
    return rates


def stack_ties(values):
    """Return a vertical offset per value so dots that would overlap stack around the row, in legend order."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    clusters, current = [], [order[0]]
    for i in order[1:]:
        if values[i] - values[current[-1]] < TIE_WIDTH:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)
    offsets = [0.0] * len(values)
    for cluster in clusters:
        for rank, i in enumerate(sorted(cluster)):  # legend order top to bottom, so a stack reads like the legend
            offsets[i] = (rank - (len(cluster) - 1) / 2) * STACK_STEP
    return offsets


def draw_plot(rates):
    """Draw one row per judge with a dot per model, and write the PNG and SVG."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    models = list(MODELS)
    for row, (judge, _) in enumerate(ROWS):
        values = [100 * rates[model][judge][0] / rates[model][judge][1] for model in models]
        if max(values) > min(values):
            ax.plot([min(values), max(values)], [row, row], color=NEUTRAL_LIGHT, linewidth=1.4, solid_capstyle="round", zorder=1)
        for model, value, offset in zip(models, values, stack_ties(values)):
            ax.scatter(value, row + offset, s=118, color=MODELS[model], edgecolor=BG, linewidth=1.8, zorder=3)  # y is inverted
        if min(values) == max(values) == 100:
            ax.text(97.4, row, "all six models", ha="right", va="center", fontsize=10.5, color=MUTED, fontstyle="italic")
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    ax.set_yticks(range(len(ROWS)), [label for _, label in ROWS])
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.tick_params(axis="x", colors=MUTED, labelsize=11, length=0, pad=8)
    ax.tick_params(axis="y", colors=INK, labelsize=12, length=0, pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    place_title(fig, "How often each judge passed each model's answers")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor=MODELS[model],
                      markeredgecolor=BG, markeredgewidth=1.5) for model in models]
    fig.legend(handles, models, loc="upper left", bbox_to_anchor=legend_anchor(fig), ncol=len(models), frameon=False,
               fontsize=11, labelcolor=TEXT, handletextpad=0.35, columnspacing=1.5, borderaxespad=0)
    place_xlabel(fig, "Pass rate")
    fig.subplots_adjust(left=0.2, right=RIGHT, top=plot_top(fig, legend=True), bottom=plot_bottom(fig))
    for extension in ("png", "svg"):
        fig.savefig(FIGURE.with_suffix("." + extension), facecolor=BG)
    plt.close(fig)


def main():
    """Count each judge's passes per model, print them, and draw the figure."""
    rates = count_passes()
    for model, judges in rates.items():
        print(model + ": " + ", ".join(f"{judge} {passed}/{scored}" for judge, (passed, scored) in judges.items()))
    with plt.rc_context(attempts_source.FONTS):
        draw_plot(rates)
    print(f"Wrote {FIGURE.relative_to(ROOT)}.png/.svg")


if __name__ == "__main__":
    main()
