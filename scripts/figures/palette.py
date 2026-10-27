"""Colour roles shared by the figure scripts, so the README figures read as one set.

Import roles instead of repeating hex values:
    from palette import BG, INK, MODELS

MODELS lists the six models in display order. Its hues were checked with the dataviz
skill's validate_palette.js (--mode light --pairs all, surface BG) and every hard gate
passes. Three hues sit just under 3:1 against BG and the Nemotron/GLM pair is in the
colour-vision warn band, which is acceptable only because every figure labels the
models directly. Re-run the validator after changing a model hue.
"""

BG = "#FBFBF9"  # figure and axes background
INK = "#1F2328"  # titles, axis labels and the strongest text
TEXT = "#3F4650"  # legend and body text
MUTED = "#6B7280"  # tick labels and secondary text
GRID = "#E4E4DE"  # gridlines and spines

PRIMARY = "#2A78D6"  # the measured thing: after, passed, confirmed correct
PROBLEM = "#CF6A3E"  # contradicted, mistaken: a terracotta one step quieter than pure orange
NEUTRAL_LIGHT = "#B4B9C0"  # before, redundant, no answer
NEUTRAL_DARK = "#4A4F57"  # a darker neutral for the same roles
UNVERIFIED = "#A9C8EE"  # nothing contradicted, but a claim could not be checked

MODELS = {  # display order; Qwen-3.5 keeps PRIMARY because it was the assistant's model
    "Qwen-3.6 35B": "#CD8431",
    "Qwen-3.5 122B": PRIMARY,
    "Nemotron 120B": "#4E7A2A",
    "Muse 30B": "#AA81D6",
    "GLM 5.3 Flash 320B": "#A3305B",
    "DeepSeek V4 Flash 284B": "#01A79B",
}

# The shared frame: every figure is WIDTH inches wide with the same outer margin, title size and offset, legend band and
# axis-label band. Only the height varies, so the figures sit at one scale in the README.
WIDTH, DPI, PAD = 11.5, 200, 0.35
TITLE_SIZE = 19
RIGHT = 1 - PAD / WIDTH


def place_title(fig, text):
    """Put the title at the shared offset from the top-left corner."""
    fig.text(PAD / WIDTH, 1 - PAD / fig.get_figheight(), text, fontsize=TITLE_SIZE, fontweight="bold", color=INK, va="top")


def legend_anchor(fig):
    """Upper-left anchor for the legend row that sits under the title."""
    return (PAD / WIDTH - 0.006, 1 - 0.9 / fig.get_figheight())


def plot_top(fig, legend=False):
    """Top of the plot area: below the title, and below the legend or column-header row when there is one."""
    return 1 - (1.55 if legend else 1.05) / fig.get_figheight()


def plot_bottom(fig, xlabel=True):
    """Bottom of the plot area, leaving room for tick labels and, when there is one, an axis label."""
    return (0.85 if xlabel else 0.45) / fig.get_figheight()


def place_xlabel(fig, text):
    """Axis label centred on the figure at the shared distance from the bottom edge."""
    fig.text(0.5, 0.3 / fig.get_figheight(), text, ha="center", va="baseline", fontsize=12, color=INK)
