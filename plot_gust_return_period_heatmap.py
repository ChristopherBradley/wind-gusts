#!/usr/bin/env python3
"""Heatmap of the wind-gust exceedance-probability table written by
gust_return_period_table.py - one figure, both sites side by side as two
panels sharing a colorbar. Rows are return periods (years), columns are
gust-speed thresholds (m/s), and each cell is the probability (%) of at
least one gust at or above that threshold occurring within that many years.

Reds mark high probability (bad - a damaging gust is likely), blues mark low
probability (good), diverging through a neutral midpoint at 50%.

Reads only that CSV - no BARRA2 file access, so it's fast and safe to
re-run/tweak.

Example:
    python plot_gust_return_period_heatmap.py \\
        --csv /scratch/xe2/cb8590/wind-gusts2/gust_exceedance_probabilities_wsgsmax_bias_197901-202601.csv
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from barra_common import VARIABLES

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

# Diverging red<->blue, neutral midpoint at 50% - high probability (bad) reads
# red, low probability (good) reads blue (ColorBrewer RdBu, reversed).
CMAP = plt.get_cmap("RdBu_r")

THRESHOLD_COL_RE = re.compile(r"p_exceed_([\d.]+)ms_pct")

# Short formula shown in the title, for the variables this script is actually
# used with (see barra_common.VARIABLES for the full derivations).
FORMULA_LABELS = {
    "wsgsmax_bias": "wsgsmax * 0.9",
    "wsgsmax": "wsgsmax * 0.9, height-corrected to 2 m",
}


def detect_variable(tag):
    return next((v for v in sorted(VARIABLES, key=len, reverse=True) if tag.startswith(v + "_")), None)


def threshold_columns(df):
    cols = [c for c in df.columns if THRESHOLD_COL_RE.match(c)]
    cols = sorted(cols, key=lambda c: float(THRESHOLD_COL_RE.match(c).group(1)))
    labels = [f"{THRESHOLD_COL_RE.match(c).group(1)} m/s" for c in cols]
    return cols, labels


def text_color_for(value, vmin=0, vmax=100):
    """White text on the dark ends of the diverging ramp (near 0% or 100%),
    dark text near the light midpoint (~50%) - computed from the actual
    cmap luminance rather than assumed, so it stays correct if CMAP changes."""
    r, g, b, _ = CMAP((value - vmin) / (vmax - vmin))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "#0b0b0b"


def plot_heatmaps(df, threshold_cols, threshold_labels, title, out_path):
    points = sorted(df["point"].unique())
    fig, axes = plt.subplots(1, len(points), figsize=(1.4 * len(threshold_cols) * len(points) + 2, 0.8 * 5 + 2), sharey=True)

    im = None
    for ax, point in zip(axes, points):
        g = df[df["point"] == point]
        point_label = g["point_label"].iloc[0]
        pivot = g.set_index("return_period_years")[threshold_cols].sort_index()
        data = pivot.values

        im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(threshold_labels)))
        ax.set_xticklabels(threshold_labels)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{t:g}" for t in pivot.index])
        ax.set_xlabel("Gust speed threshold")
        ax.set_title(point_label)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color_for(value), fontsize=9)

    axes[0].set_ylabel("Return period (years)")
    fig.suptitle(title)

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Probability (%)")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    threshold_cols, threshold_labels = threshold_columns(df)
    tag = csv_path.stem
    if tag.startswith("gust_exceedance_probabilities_"):
        tag = tag[len("gust_exceedance_probabilities_"):]

    variable = detect_variable(tag)
    formula = FORMULA_LABELS.get(variable)
    title = "Probability of exceeding gust speed within N years in August to November"
    if formula:
        title += f" ({formula})"

    out_path = out_dir / f"gust_exceedance_heatmap_{tag}.png"
    plot_heatmaps(df, threshold_cols, threshold_labels, title, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.csv), Path(args.out_dir))
