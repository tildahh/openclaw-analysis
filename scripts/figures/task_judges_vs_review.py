"""Plot each Phoenix built-in judge against the correctness review of the task answers.

One point per judge: the share of the answers the review found contradicted that the judge flagged (x), against the share
of the answers the review confirmed correct that the judge passed (y). The ideal judge sits in the top-right corner.
Judges are coloured by what they are built to check, because only two of them compare the answer with its tool results.
Reads results/task-judge-agreement.json (evals/compare_judges_with_review.py) and writes figures/task-judges-vs-review.png:

    python3 scripts/figures/task_judges_vs_review.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from palette import BG, GRID, INK, MUTED, NEUTRAL_LIGHT, PRIMARY, TEXT, DPI, RIGHT, WIDTH, legend_anchor, place_title, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
AGREEMENT = ROOT / "results" / "task-judge-agreement.json"

SOURCES, OTHER = PRIMARY, NEUTRAL_LIGHT  # checks the answer against its tool results, or something else
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}

# key, label, checks the answer against its tool results?, label placement (ha, dx, dy in points)
JUDGES = [
    ("faithfulness", "Faithfulness", True, ("right", -12, 0)),
    ("tool_response_handling", "Tool response handling", True, ("right", -12, 0)),
    ("task_completion", "Task completion", False, ("left", 12, 0)),
    ("tool_selection", "Tool selection", False, ("left", 12, 0)),
    ("retrieval_relevance", "Retrieval relevance", False, ("left", 12, 0)),
]


def draw_chart(agreement, out_path):
    """Draw one labelled point per judge, with the counts behind each share and the ideal corner marked."""
    fig, ax = plt.subplots(figsize=(WIDTH, 6.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for key, name, checks_sources, (ha, dx, dy) in JUDGES:
        a = agreement[key]
        x = 100 * a["failing_flagged"] / a["failing_total"]
        y = 100 * a["passing_passed"] / a["passing_total"]
        ax.scatter(x, y, s=150, color=SOURCES if checks_sources else OTHER, edgecolor=BG, linewidth=1.2, zorder=4)
        ax.annotate(name, (x, y), xytext=(dx, dy + 7), textcoords="offset points", ha=ha, va="center",
                    fontsize=12, fontweight="bold", color=INK, zorder=5)
        ax.annotate(f"{a['failing_flagged']}/{a['failing_total']} flagged · {a['passing_passed']}/{a['passing_total']} passed",
                    (x, y), xytext=(dx, dy - 9), textcoords="offset points", ha=ha, va="center", fontsize=10.5,
                    color=TEXT, zorder=5)

    ax.scatter(100, 100, s=150, facecolor="none", edgecolor=MUTED, linewidth=1.4, linestyle=(0, (2, 2)), zorder=3)
    ax.annotate("Ideal judge", (100, 100), xytext=(-14, 0), textcoords="offset points", ha="right", va="center",
                fontsize=11, color=MUTED)

    ticks = range(0, 101, 25)
    ax.set_xlim(-4, 106)
    ax.set_ylim(20, 108)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=10, color=MUTED)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in (25, 50, 75, 100)], fontsize=10, color=MUTED)
    ax.set_xlabel("Contradicted answers the judge flagged", fontsize=12, color=INK, labelpad=10)
    ax.set_ylabel("Correct answers the judge passed", fontsize=12, color=INK, labelpad=10)
    ax.grid(color=GRID, linewidth=0.75, zorder=1)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    place_title(fig, "Only source-checking judges caught nearly all contradicted answers")
    marker = dict(marker="o", linestyle="", markersize=10, markeredgecolor=BG)
    fig.legend(handles=[Line2D([], [], color=SOURCES, label="Checks the answer against its tool results", **marker),
                        Line2D([], [], color=OTHER, label="Built to check something else", **marker)],
               loc="upper left", bbox_to_anchor=legend_anchor(fig), ncol=2, frameon=False, fontsize=11,
               labelcolor=TEXT, handletextpad=0.4, columnspacing=2)
    fig.subplots_adjust(left=0.1, right=RIGHT, top=plot_top(fig, legend=True), bottom=plot_bottom(fig, xlabel=True) + 0.02)
    fig.savefig(out_path, facecolor=BG)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    with plt.rc_context(FONTS):
        draw_chart(json.loads(AGREEMENT.read_text())["agreement"], ROOT / "figures" / "task-judges-vs-review.png")
