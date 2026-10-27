"""Plot how each Phoenix built-in judge compares with the correctness review of the task answers.

Two bars per judge: the share of the answers the review found wrong that the judge flagged, and the share of the answers
it confirmed correct that the judge passed. Reads results/task-judge-agreement.json (evals/compare_judges_with_review.py) and writes
task-judges-vs-review.png next to this script:

    python3 scripts/figures/task_judges_vs_review.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from palette import BG, GRID, INK, MUTED, PRIMARY, PROBLEM, TEXT, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
AGREEMENT = ROOT / "results" / "task-judge-agreement.json"

FLAGGED, PASSED = PROBLEM, PRIMARY  # wrong answers the judge caught, correct answers it passed
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}

ROWS = [
    ("task_completion", "Task completion"),
    ("retrieval_relevance", "Retrieval relevance"),
    ("tool_selection", "Tool selection"),
    ("faithfulness", "Faithfulness"),
    ("tool_response_handling", "Tool response handling"),
]
BAR, OFFSET = 0.3, 0.17  # bar height and distance from the row's center, leaving a thin gap between the pair


def draw_chart(agreement, out_path):
    """Draw one pair of bars per judge, labeled with the counts behind each share."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    rows = sorted(ROWS, key=lambda row: agreement[row[0]]["failing_flagged"] / agreement[row[0]]["failing_total"])  # lenient to strict
    for y, (key, name) in enumerate(rows):
        a = agreement[key]
        bars = [(y - OFFSET, FLAGGED, a["failing_flagged"], a["failing_total"]),
                (y + OFFSET, PASSED, a["passing_passed"], a["passing_total"])]
        for center, color, count, total in bars:
            share = 100 * count / total
            ax.barh(center, share, height=BAR, color=color, zorder=3)
            ax.text(share + 1.2, center, f"{count}/{total}", va="center", ha="left", fontsize=11, color=TEXT)
        ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=INK)

    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 25)], fontsize=10, color=MUTED)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.75, zorder=1)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    place_title(fig, "Contradicted answers flagged and correct answers passed, by judge")
    fig.legend(handles=[Patch(color=FLAGGED, label="Contradicted answers flagged"),
                        Patch(color=PASSED, label="Confirmed-correct answers passed")],
               loc="upper left", bbox_to_anchor=legend_anchor(fig), ncol=2, frameon=False, fontsize=11,
               labelcolor=TEXT, handlelength=1.2, handleheight=0.8, columnspacing=2)
    fig.subplots_adjust(left=0.215, right=RIGHT, top=plot_top(fig, legend=True), bottom=plot_bottom(fig, xlabel=False))
    fig.savefig(out_path, facecolor=BG)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    with plt.rc_context(FONTS):
        draw_chart(json.loads(AGREEMENT.read_text())["agreement"], ROOT / "figures" / "task-judges-vs-review.png")
