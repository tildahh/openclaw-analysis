"""Draw the opening figure: four before/after comparisons of one day of heartbeats and the day after the changes.

Before is the five heartbeats that shared the whole week's chat (period shared_whole_day). After is the next
day's 23 heartbeats under fresh sessions, the rewritten silence rule and live-source checks. Outcomes use my
own labels, not the judge's.

Sources, all under results/: judged_heartbeats.csv and labels.csv for the before day (first-call tokens and
labels), labels-production.csv and production-heartbeats.json for the after day (labels,
first-call tokens, NO_REPLY-appended messages). The explanatory lines live in the README next to the figure, not in the image. Writes figures/heartbeat-before-after.png and .svg.
Run from the repo root with any Python that has matplotlib:
    python scripts/figures/heartbeat_before_after.py
"""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import BG, GRID, INK, MUTED, NEUTRAL_LIGHT, PRIMARY, TEXT, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "heartbeat-before-after"
BEFORE, AFTER = NEUTRAL_LIGHT, PRIMARY
SIZE = 11.5
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}
WORTH = {"useful", "somewhat_useful"}
MESSAGE = WORTH | {"redundant"}  # labels for heartbeats that sent a message


def load_numbers():
    """Read both days from the saved results and return the four before/after pairs."""
    labels = {r["trace_id"]: r["decision"] for r in csv.DictReader(open(ROOT / "results" / "labels.csv"))}
    before = [r for r in csv.DictReader(open(ROOT / "results" / "judged_heartbeats.csv")) if r["period"] == "shared_whole_day"]
    after_labels = list(csv.DictReader(open(ROOT / "results" / "labels-production.csv")))
    after = json.load(open(ROOT / "results" / "production-heartbeats.json"))
    assert len(before) == 5 and len(after_labels) == 23 and after["heartbeat_count"] == 23
    before_decisions = [labels[r["trace_id"]] for r in before]
    after_decisions = [r["label"] for r in after_labels]
    return {
        "n": (len(before), len(after_labels)),
        "tokens": (statistics.median(float(r["tokens_first_call"]) for r in before), after["median_first_call_input_tokens"]),
        "leaked": (sum(bool(r["report_plus_no_reply"]) for r in after["baseline"]), after["report_plus_no_reply"]),
        "worth": (sum(d in WORTH for d in before_decisions), sum(d in WORTH for d in after_decisions)),
        "messages": (sum(d in MESSAGE for d in before_decisions), sum(d in MESSAGE for d in after_decisions)),
        "silent": (before_decisions.count("correct_silence"), after_decisions.count("correct_silence")),
    }


def draw(numbers):
    """Four panels in a 2x2 block, each with a Before bar above an After bar and the value written at the bar's end."""
    plt.rcParams.update(FONTS)
    n_before, n_after = numbers["n"]
    panels = [
        ("Context at the start of each run", numbers["tokens"], lambda v: f"{v / 1000:.0f}k tokens", None),
        ("Messages delivered with NO_REPLY attached", numbers["leaked"], None, numbers["messages"]),
        ("Heartbeats worth hearing", numbers["worth"], None, numbers["n"]),
        ("Correct silences", numbers["silent"], None, numbers["n"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.9), dpi=DPI, facecolor=BG)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.085, right=RIGHT, top=plot_top(fig, legend=True), bottom=plot_bottom(fig, xlabel=False), hspace=0.62, wspace=0.26)
    for i, (ax, (title, (b, a), fmt, totals)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(BG)
        if fmt is None:  # counts become shares of the panel's totals, labelled as "n/N"
            n_b, n_a = totals
            vals = [b / n_b, a / n_a]
            texts = [f"{b}/{n_b}", f"{a}/{n_a}"]
            limit = 1.0
        else:
            vals = [b, a]
            texts = [fmt(b), fmt(a)]
            limit = max(vals)
        for y, (v, t, colour) in enumerate(zip(vals, texts, (BEFORE, AFTER))):
            ax.barh(y, v, height=0.56, color=colour, edgecolor=BG, linewidth=1)
            ax.text(v + limit * 0.03, y, t, va="center", ha="left", fontsize=SIZE, color=INK,
                    fontweight="bold" if y == 1 else "normal")
        ax.set_xlim(0, limit * 1.3)
        ax.set_ylim(1.6, -0.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Before", "After"] if i % 2 == 0 else ["", ""], fontsize=SIZE, color=TEXT)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.set_title(title, loc="left", fontsize=SIZE + 0.5, color=INK, fontweight="bold", pad=10)
    place_title(fig, "Heartbeats before and after the changes")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor=BG)
    fig.savefig(OUT.with_suffix(".svg"), facecolor=BG)


if __name__ == "__main__":
    numbers = load_numbers()
    print(json.dumps(numbers, indent=1))
    draw(numbers)
    print("wrote", OUT.with_suffix(".png"))
