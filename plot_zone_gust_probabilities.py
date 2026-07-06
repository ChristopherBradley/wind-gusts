#!/usr/bin/env python3
"""Annual wind-gust exceedance-probability curves per GRDC cropping zone, in
the style of the project's "Wind gust probabilities" figure (Papers/): for a
range of gust-speed thresholds, the empirical annual probability that the
Aug-Nov (growing-season) maximum gust in a zone equals or exceeds that speed.

"Annual probability" here is the fraction of years in the record whose Aug-Nov
seasonal-maximum gust is >= the threshold - i.e. an empirical annual-maximum
exceedance curve, no distribution fitted. It runs from 1.0 at low speeds (every
year exceeds them) down to 0.0 at high speeds.

Input is the wide daily CSV from extract_zone_gust_history.py. `--stat` picks
which per-zone statistic's daily time series to build the curve from:
  * median (default) - the "median timeseries", the typical gust across a zone.
  * max              - the windiest pixel in the zone each day.

Outputs (per stat), written to --out-dir:
  * gust_zone_exceedance_prob_<stat>_<tag>_m08-11.csv - the table: one row per
    threshold, one column per zone, values = annual probability.
  * gust_zone_exceedance_prob_<stat>_<tag>_m08-11.png - combined 6x3 panel of
    all 18 zones, shared axes.
  * gust_zone_exceedance_<stat>_<tag>_m08-11/<zone>.png - one single-zone
    figure per zone.

Reads only the CSV - no BARRA2 access, so it is fast and safe to re-run/tweak.

Example:
    python plot_zone_gust_probabilities.py --stat median \\
        --csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from barra_common import VARIABLES

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

# Short formula shown in the figure title (see barra_common.VARIABLES).
FORMULA_LABELS = {
    "wsgsmax_bias": "wsgsmax * 0.9",
    "wsgsmax": "wsgsmax * 0.9, height-corrected to 2 m",
}

STAT_LABELS = {"median": "zone median", "max": "zone maximum",
               "mean": "zone mean", "p75": "zone p75", "p90": "zone p90"}

LINE_COLOR = "#2b6cb0"
MARKER_FACE = "#e9d43a"


def detect_variable(tag):
    return next((v for v in sorted(VARIABLES, key=len, reverse=True) if tag.startswith(v + "_")), None)


def safe_name(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def seasonal_annual_max(series, start_month, end_month):
    """Aug-Nov maximum per year for a daily series indexed by date. Only years
    with all four season months present are kept."""
    s = series[series.index.month.to_series(index=series.index).between(start_month, end_month)]
    by_year = s.groupby(s.index.year)
    n_months = by_year.apply(lambda g: g.index.month.nunique())
    complete = n_months[n_months == (end_month - start_month + 1)].index
    return by_year.max().loc[complete]


def exceedance_curve(annual_max, thresholds):
    vals = annual_max.values[:, None]  # (n_years, 1)
    return (vals >= thresholds[None, :]).mean(axis=0)  # fraction of years >= each threshold


def plot_single(ax, thresholds, probs, title, xlim=None):
    ax.plot(thresholds, probs, "-o", color=LINE_COLOR, markerfacecolor=MARKER_FACE,
            markeredgecolor=LINE_COLOR, markersize=4, linewidth=1.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(*(xlim if xlim is not None else (thresholds[0], thresholds[-1])))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(title, fontsize=10)
    # Every panel gets its own axis labels - the 6x3 grid is too wide to read
    # a shared label off the far edge.
    ax.set_xlabel("Wind gust speed (m/s)")
    ax.set_ylabel("Annual probability")


def main(csv_path, out_dir, stat, start_month, end_month, xmin, xmax):
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    tag = csv_path.stem
    if tag.startswith("gust_zone_summary_"):
        tag = tag[len("gust_zone_summary_"):]
    variable = detect_variable(tag)
    formula = FORMULA_LABELS.get(variable, variable)
    season_tag = f"{stat}_{tag}_m{start_month:02d}-{end_month:02d}"

    suffix = f"_{stat}"
    zone_cols = [c for c in df.columns if c.endswith(suffix)]
    if not zone_cols:
        raise SystemExit(f"No columns ending in {suffix!r} in {csv_path}")
    zones = [c[: -len(suffix)] for c in zone_cols]

    # Per-zone Aug-Nov annual maxima, then a shared threshold grid.
    annual_maxes = {z: seasonal_annual_max(df[col], start_month, end_month) for z, col in zip(zones, zone_cols)}
    overall_max = max(am.max() for am in annual_maxes.values())
    x_hi = int(math.ceil(overall_max / 5.0) * 5)
    thresholds = np.arange(0, x_hi + 1, 1.0)

    curves = {z: exceedance_curve(am, thresholds) for z, am in annual_maxes.items()}
    n_years = {z: len(am) for z, am in annual_maxes.items()}
    xlim = (xmin if xmin is not None else float(thresholds[0]),
            xmax if xmax is not None else float(thresholds[-1]))

    # --- Table CSV: rows = threshold, cols = zone ---------------------------
    table = pd.DataFrame(curves, index=pd.Index(thresholds, name="gust_speed_m_s"))
    table_csv = out_dir / f"gust_zone_exceedance_prob_{season_tag}.csv"
    table.round(4).to_csv(table_csv)
    print(f"Wrote {table_csv}")

    stat_label = STAT_LABELS.get(stat, stat)
    supt = (f"Annual probability of the Aug-Nov maximum gust ({stat_label}) exceeding a speed "
            f"per GRDC cropping zone\n({formula})")

    # --- Combined 6x3 panel figure -----------------------------------------
    # Independent axes (no sharex/sharey) so every panel carries its own
    # tick labels and axis titles.
    nrows, ncols = 6, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 20))
    for idx, z in enumerate(zones):
        ax = axes[divmod(idx, ncols)]
        plot_single(ax, thresholds, curves[z], z, xlim=xlim)
    for idx in range(len(zones), nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)
    fig.suptitle(supt, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    combined_png = out_dir / f"gust_zone_exceedance_prob_{season_tag}.png"
    fig.savefig(combined_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {combined_png}")

    # --- One figure per zone -----------------------------------------------
    per_zone_dir = out_dir / f"gust_zone_exceedance_{season_tag}"
    per_zone_dir.mkdir(parents=True, exist_ok=True)
    for z in zones:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_single(ax, thresholds, curves[z], f"{z} ({stat_label})", xlim=xlim)
        fig.suptitle(f"Annual gust exceedance probability, Aug-Nov  ({formula})", fontsize=11)
        fig.tight_layout()
        fig.savefig(per_zone_dir / f"{safe_name(z)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {len(zones)} per-zone figures to {per_zone_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--stat", default="median", choices=list(STAT_LABELS))
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--xmin", type=float, default=None, help="Truncate the gust-speed axis at this lower bound (m/s)")
    parser.add_argument("--xmax", type=float, default=None, help="Truncate the gust-speed axis at this upper bound (m/s)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.csv), Path(args.out_dir), args.stat, args.start_month, args.end_month, args.xmin, args.xmax)
