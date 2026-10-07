"""Shared matplotlib style for the notebooks and evidence figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#2a78d6"     # our model / actual outcome
ORANGE = "#eb6834"   # the vendor bot
AQUA = "#1baf7a"
GREY = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d6d5d0",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "figure.dpi": 110,
    })


def heatmap(ax, table, title, fmt="{:.0f}", cmap="Blues", normalize_rows=False):
    """Annotated heatmap of a crosstab (rows = truth/first, cols = other)."""
    data = table.div(table.sum(axis=1), axis=0) if normalize_rows else table
    ax.imshow(data.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    vmax = data.values.max()
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.55 else INK)
    ax.set_title(title, loc="left")
