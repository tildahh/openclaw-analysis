"""Plot all task runtimes from results/model-completion-time.json.

Run: python scripts/figures/runtime_distribution.py
Uses the same matched attempts as the model scatter plot without new trials.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from palette import BG, GRID, INK, MODELS, MUTED, DPI, RIGHT, WIDTH, legend_anchor, place_title, place_xlabel, plot_bottom, plot_top

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/model-completion-time.json"
FIGURE = ROOT / "figures/runtime-distribution"
FONTS = {"font.family": "sans-serif",
         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}
POINT_SIZE = 30


def collect_runtimes(data):
    """Validate the matched population and retain recorded timeouts in each model's runtimes."""
    attempts = data["attempts"]
    assert len(attempts) == 162, "Expected 162 matched attempts"
    assert len({(a["source_experiment"], a["source_run_id"]) for a in attempts}) == 162, "Duplicate attempt"
    assert set(a["model"] for a in attempts) == set(MODELS), "Unexpected model"
    summary = {row["model"]: row for row in data["models"]}
    groups = {}
    for model in MODELS:
        rows = sorted((a for a in attempts if a["model"] == model), key=lambda a: a["native_seconds"])
        assert len(rows) == 27, f"Expected 27 attempts for {model}"
        assert Counter(a["case"] for a in rows) == Counter({case: 3 for case in summary[model]["task_means"]})
        assert len({(a["case"], a["pass"]) for a in rows}) == 27
        assert {a["pass"] for a in rows} == {1, 2, 3}, "Unexpected pass"
        for row in rows:
            assert math.isfinite(row["native_seconds"]) and row["native_seconds"] > 0, "Missing or invalid runtime"
            assert row["execution_label"] in {"COMPLETED", "TIMEOUT"}, "Unrecognized execution outcome"
            assert bool(row["error"]) == (row["execution_label"] == "TIMEOUT"), "Unresolved execution error"
            if row["execution_label"] == "TIMEOUT":
                assert "timed out" in row["error"].lower(), "Timeout lacks recorded evidence"
        minutes = [a["native_seconds"] / 60 for a in rows]
        assert math.isclose(statistics.median(minutes), summary[model]["median_minutes"]), "Median differs from scatter plot"
        groups[model] = {"rows": rows, "minutes": minutes, "median": statistics.median(minutes)}
    return groups


def spread_points(ax, minutes, row):
    """Separate nearby points vertically while preserving every recorded runtime on the x-axis."""
    gap = (math.sqrt(POINT_SIZE) + 1) * ax.figure.dpi / 72 + 2
    placed, offsets = [], []
    row_height = abs(ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    for minute in minutes:
        x = ax.transData.transform((minute, row))[0]
        neighbors = [(px, py) for px, py in placed if abs(x - px) < gap]
        candidates = [0.0]
        for px, py in neighbors:
            separation = math.sqrt(gap * gap - (x - px) ** 2)
            candidates.extend([py - separation, py + separation])
        y = next(candidate for candidate in sorted(candidates, key=lambda value: (abs(value), value))
                 if all(math.hypot(x - px, candidate - py) >= gap - 1e-6 for px, py in neighbors))
        assert abs(y) < row_height * .4, f"Points overflow row {row}: {abs(y):.1f} > {row_height * .4:.1f} pixels"
        placed.append((x, y))
        offsets.append(row + y / row_height)
    return offsets


def draw_distribution(groups):
    """Draw all 162 attempts with model colors, median ticks, and distinct timeout markers."""
    fig, ax = plt.subplots(figsize=(WIDTH, 7.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 21.4)
    ax.set_ylim(5.6, -.6)
    ax.set_xticks([0, 5, 10, 15, 20])
    ordered = dict(sorted(groups.items(), key=lambda item: item[1]["median"]))  # fastest median at the top
    ax.set_yticks(range(6), list(ordered), fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", length=0, pad=12, colors=MUTED)
    for label, model in zip(ax.get_yticklabels(), ordered):
        label.set_color(MODELS[model])
    ax.grid(axis="x", color=GRID, linewidth=.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    place_xlabel(fig, "Time per task (minutes)")
    fig.subplots_adjust(left=.215, right=.905, top=plot_top(fig, legend=True), bottom=plot_bottom(fig))
    fig.canvas.draw()
    for y, (model, group) in enumerate(ordered.items()):
        minutes = group["minutes"]
        offsets = spread_points(ax, minutes, y)
        ax.vlines(group["median"], y - .44, y + .44, color=INK, linewidth=2.1, zorder=2)
        ax.text(1.025, y, f"{group['median']:.2f} min", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10.5, color=INK)
        for minute, offset, attempt in zip(minutes, offsets, group["rows"]):
            if attempt["execution_label"] == "TIMEOUT":
                ax.scatter(minute, offset, s=75, marker="X", color=MODELS[model], edgecolor=BG, linewidth=.8, zorder=4)
                ax.annotate(f"{minute:.2f} min", (minute, offset), xytext=(-8, 17), textcoords="offset points",
                            ha="right", fontsize=10, color=INK)
            else:
                ax.scatter(minute, offset, s=POINT_SIZE, color=MODELS[model], edgecolor=BG, linewidth=.6, zorder=3)
    ax.text(1.025, 1.045, "Median", transform=ax.transAxes, fontsize=10.5, color=MUTED)
    place_title(fig, "Task runtime distribution by model")
    fig.legend(handles=[Line2D([], [], marker="o", color=MUTED, linestyle="none", label="Attempt", markersize=6),
                        Line2D([], [], marker="|", color=INK, linestyle="none", label="Median", markersize=13, markeredgewidth=2),
                        Line2D([], [], marker="X", color=MUTED, linestyle="none", label="Timeout", markersize=8)],
               loc="upper left", bbox_to_anchor=legend_anchor(fig), ncol=3, frameon=False, fontsize=11,
               labelcolor=INK, handletextpad=.5, columnspacing=2)
    for extension in ("png", "svg"):
        fig.savefig(FIGURE.with_suffix("." + extension), facecolor=BG)
    plt.close(fig)


def main():
    """Verify source hashes, render runtimes, and report the plotted populations and medians."""
    data = json.loads(SOURCE.read_text())
    for name, digest in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Saved source changed: {name}"
    groups = collect_runtimes(data)
    with plt.rc_context(FONTS):
        draw_distribution(groups)
    for model, group in groups.items():
        timeouts = sum(row["execution_label"] == "TIMEOUT" for row in group["rows"])
        print(f"{model}: {len(group['rows'])} attempts, median {group['median']:.6f} min, {timeouts} timeouts")
    print("Wrote", FIGURE.with_suffix(".png"), "and .svg")


if __name__ == "__main__":
    main()
