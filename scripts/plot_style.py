"""
plot_style.py — Shared matplotlib style for all pipeline figures.
Import and call apply() at the top of any figure-generating script.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

REGION_COLORS = {
    "Arizona": "#4e79a7",
    "Georgia": "#f28e2b",
    "Ohio":    "#59a14f",
    "Ontario": "#e15759",
}

CAT_COLORS = {
    "Structure": "#4e79a7",
    "Traffic":   "#f28e2b",
    "Climate":   "#59a14f",
}


def apply():
    plt.rcParams.update({
        # Font
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.titlesize":   13,
        # Layout
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "figure.autolayout":  False,
        # Axes appearance
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "axes.axisbelow":     True,
        # Lines and markers
        "lines.linewidth":    2.0,
        "patch.edgecolor":    "white",
        "patch.linewidth":    0.5,
        # Colours
        "axes.prop_cycle":    mpl.cycler(color=[
            "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
            "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
        ]),
    })
