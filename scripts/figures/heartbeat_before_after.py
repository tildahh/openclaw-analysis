"""Draw what each heartbeat did before and after the changes: one square per heartbeat, by my label.

Before is the five heartbeats that shared the whole week's chat (period shared_whole_day). After is the next day's
23 heartbeats under fresh sessions, the rewritten silence rule and live-source checks. The labels are mine, not the
judge's. Sources, all under results/: judged_heartbeats.csv and labels.csv for the before day, labels-production.csv
for the after day. Writes figures/heartbeat-before-after.png and .svg.
Run from the repo root with any Python that has matplotlib:
    python scripts/figures/heartbeat_before_after.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from palette import BG, INK, MUTED, NEUTRAL_LIGHT, PRIMARY, PROBLEM, TEXT, DPI, WIDTH, legend_anchor, place_title

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "heartbeat-before-after"
SIZE = 11.5
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}
# Each outcome in the order it is drawn along a row: name, the labels it covers, fill and edge.
# A silence sends nothing, so it is hollow; a missed reminder is hollow in the problem colour.
OUTCOMES = [
    ("Worth hearing", {"useful", "somewhat_useful"}, PRIMARY, PRIMARY),
    ("Correct silence", {"correct_silence"}, BG, PRIMARY),
    ("Unnecessary message", {"redundant"}, NEUTRAL_LIGHT, NEUTRAL_LIGHT),
    ("Missed reminder", {"missed"}, BG, PROBLEM),
]
HEIGHT = 3.35                     # inches; every position below is in inches from the bottom-left corner
SQUARE, GAP, GROUP_GAP = 0.29, 0.05, 0.22
LEFT = 1.0                        # where the squares start; the row names end just before it
ROWS = {"Before": 1.42, "After": 0.42}  # bottom edge of each row of squares


def load_rows():
    """Read my labels for both days and count each outcome."""
    labels = {r["trace_id"]: r["decision"] for r in csv.DictReader(open(ROOT / "results" / "labels.csv"))}
    before = [labels[r["trace_id"]] for r in csv.DictReader(open(ROOT / "results" / "judged_heartbeats.csv"))
              if r["period"] == "shared_whole_day"]
    after = [r["label"] for r in csv.DictReader(open(ROOT / "results" / "labels-production.csv"))]
    assert len(before) == 5 and len(after) == 23
    rows = {}
    for name, decisions in (("Before", before), ("After", after)):
        counts = [sum(d in covered for d in decisions) for _, covered, _, _ in OUTCOMES]
        assert sum(counts) == len(decisions), (name, "a label outside the four outcomes", sorted(set(decisions)))
        rows[name] = counts
    return rows


def square(ax, x, y, fill, edge):
    """One rounded square; hollow squares are inset by half their outline so every square has the same footprint."""
    width = 1.6 / 72 if fill == BG else 0
    ax.add_patch(FancyBboxPatch((x + width / 2, y + width / 2), SQUARE - width, SQUARE - width,
                                boxstyle="round,pad=0,rounding_size=0.035", facecolor=fill, edgecolor=edge,
                                linewidth=1.6 if width else 0))


def draw(rows):
    """One row of squares per day, grouped by outcome, with each group's count above it and the row total after it."""
    plt.rcParams.update(FONTS)
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_axis_off()
    for row, y in ROWS.items():
        ax.text(LEFT - 0.16, y + SQUARE / 2, row, ha="right", va="center", fontsize=SIZE, color=TEXT)
        x = LEFT
        for (name, _, fill, edge), count in zip(OUTCOMES, rows[row]):
            if not count:
                continue
            for i in range(count):
                square(ax, x + i * (SQUARE + GAP), y, fill, edge)
            width = count * (SQUARE + GAP) - GAP
            ax.text(x + width / 2, y + SQUARE + 0.1, str(count), ha="center", va="bottom", fontsize=SIZE, color=INK,
                    fontweight="bold" if row == "After" else "normal")
            x += width + GROUP_GAP
        ax.text(x - GROUP_GAP + 0.18, y + SQUARE / 2, f"{sum(rows[row])} heartbeats", ha="left", va="center",
                fontsize=SIZE - 1, color=MUTED)
    handles = [Line2D([], [], linestyle="none", marker="s", markersize=11, markerfacecolor=fill, markeredgecolor=edge,
                      markeredgewidth=1.6 if fill == BG else 0) for _, _, fill, edge in OUTCOMES]
    names = [name if any(rows[r][i] for r in rows) else f"{name} (none)" for i, (name, *_ ) in enumerate(OUTCOMES)]
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=legend_anchor(fig), ncol=4, frameon=False,
               fontsize=SIZE, labelcolor=TEXT, handletextpad=0.4, columnspacing=1.6, borderaxespad=0)
    place_title(fig, "Heartbeats before and after the changes")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor=BG)
    fig.savefig(OUT.with_suffix(".svg"), facecolor=BG)


if __name__ == "__main__":
    rows = load_rows()
    print(json.dumps(rows))
    draw(rows)
    print("wrote", OUT.with_suffix(".png"))
