"""Plot each model's 27 task attempts by the correctness review's verdict, as one stacked bar per model.

Confirmed correct: every material claim checked out. Unverifiable: nothing contradicted, but a claim couldn't be
checked. Contradicted: the sources contradict a claim. No answer: a timeout or a reply that was only tool-call markup.
Reads results/task-judge-agreement.json (evals/compare_judges_with_review.py) and writes correctness-by-model.png next
to this script:

    python3 scripts/figures/correctness_by_model.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from palette import BG, INK, MUTED, NEUTRAL_LIGHT, PRIMARY, PROBLEM, TEXT, UNVERIFIED, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}
AGREEMENT = ROOT / "results" / "task-judge-agreement.json"

SEGMENTS = [  # (outcome in the data, legend label, fill, label ink)
    ("PASS", "Confirmed correct", PRIMARY, "#FFFFFF"),
    ("UNSURE", "Could not fully verify", UNVERIFIED, INK),
    ("FAIL", "Contradicted by sources", PROBLEM, "#FFFFFF"),
    ("no answer", "No answer", NEUTRAL_LIGHT, INK),
]


def draw_chart(rows, out_path):
    """Draw one stacked bar of 27 attempts per model, each segment labeled with its count."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for y, row in enumerate(rows):
        left = 0
        for key, _, fill, ink in SEGMENTS:
            count = row["outcomes"][key]
            if not count:
                continue
            ax.barh(y, count, left=left, height=0.62, color=fill, edgecolor=BG, linewidth=2, zorder=3)
            ax.text(left + count / 2, y, str(count), ha="center", va="center", fontsize=11, color=ink, zorder=4)
            left += count
        ax.text(-0.015, y, row["model"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 27)
    ax.set_xticks([0, 9, 18, 27])
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    place_title(fig, "Each model's 27 attempts, by the correctness review's verdict")
    fig.legend(handles=[Patch(color=fill, label=label) for _, label, fill, _ in SEGMENTS], loc="upper left",
               bbox_to_anchor=legend_anchor(fig), ncol=4, frameon=False, fontsize=10.5, labelcolor=TEXT, handlelength=1.2)
    fig.subplots_adjust(left=0.22, right=RIGHT, top=plot_top(fig, legend=True), bottom=plot_bottom(fig, xlabel=False))
    fig.savefig(out_path, facecolor=BG)
    print(f"wrote {out_path}")


def main():
    """Read the per-model outcome counts and draw the chart."""
    rows = json.loads(AGREEMENT.read_text())["by_model"]
    with plt.rc_context(FONTS):
        draw_chart(rows, ROOT / "figures" / "correctness-by-model.png")


if __name__ == "__main__":
    main()
