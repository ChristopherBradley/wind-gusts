#!/usr/bin/env python3
"""Plot the full-history gust climatology at the two paddock points, from a
CSV written by extract_gust_history.py. Reads only that CSV - never touches
the BARRA2 files - so the plot can be tweaked without re-running the (slow)
extraction.

Three plots are written (the daily series is too dense to read as a 46-year
line plot, so all of them summarise it to at least monthly resolution, which
preserves the extreme-value signal that a mean would wash out):
  - monthly maximum, full year, every month;
  - monthly maximum, but only the Aug-Nov growing season months plotted each
    year (`--start-month`/`--end-month`, matching this project's stem-lodging
    risk window - see README) - same monthly resolution as the first plot,
    just with the off-season months blanked out so the line breaks between
    seasons instead of running Nov straight into next August. Marked with a
    dotted `--threshold` line (default 22 m/s, the Pinera-Chavez stem-lodging
    gust);
  - seasonal (Aug-Nov) maximum, one value per year, not the full calendar
    year.

Example:
    python plot_gust_climatology.py \\
        --csv /scratch/xe2/cb8590/wind-gusts2/gust_history_wsgsmax_bias_197901-202601.csv
"""
import argparse
import calendar
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from barra_common import VARIABLES

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

POINT_LABELS = {
    "paddock_2024": "Site A (-35.0504, 147.3188)",
    "paddock_2025": "Site B (-35.0259, 147.3547)",
}
COLORS = {"paddock_2024": "tab:blue", "paddock_2025": "tab:orange"}


def tag_and_label(csv_path, variable):
    tag = csv_path.stem
    if tag.startswith("gust_history_"):
        tag = tag[len("gust_history_"):]
    if variable is None:
        variable = next((v for v in sorted(VARIABLES, key=len, reverse=True) if tag.startswith(v + "_")), None)
    label = VARIABLES[variable]["long_name"] if variable in VARIABLES else tag
    return tag, label


def title_label(label):
    """Drop the parenthetical qualifier (e.g. "(bias-adjusted only, still at
    10 m)") from a long_name for use in a plot title - too much detail for a
    title, still useful in the README/CLI output where `label` is used as-is."""
    return re.sub(r"\s*\([^)]*\)", "", label).strip()


def month_range_label(start_month, end_month):
    """"August to November" - full month names for a plot title, vs. the
    compact zero-padded "m08-11" kept in filenames/tags for sortability."""
    return f"{calendar.month_name[start_month]} to {calendar.month_name[end_month]}"


def monthly_max(df):
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["point", "month"], as_index=False)["gust_m_s"].max()


def plot_monthly_max(monthly, label, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for point, g in monthly.groupby("point"):
        g = g.sort_values("month")
        ax.plot(g["month"], g["gust_m_s"], linewidth=0.8, label=POINT_LABELS.get(point, point), color=COLORS.get(point))
    ax.set_xlabel("Month")
    ax.set_ylabel("m/s")
    ax.set_title(f"{title_label(label)} - monthly maximum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def monthly_max_in_season(monthly, start_month, end_month, connect=False):
    """Monthly-max series restricted to the season's months. By default
    (connect=False) the off-season months are blanked out (NaN) so the
    plotted line breaks between each year's season instead of running a
    straight line from Nov to the following Aug; with connect=True those
    rows are dropped instead, so the line instead joins straight across the
    gap - the same "connect what data exists" convention plot_seasonal_max
    already uses for its one-point-per-year series."""
    df = monthly.copy()
    in_season = df["month"].dt.month.between(start_month, end_month)
    if connect:
        return df[in_season]
    df.loc[~in_season, "gust_m_s"] = float("nan")
    return df


def plot_monthly_max_season(monthly_masked, label, start_month, end_month, threshold, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for point, g in monthly_masked.groupby("point"):
        g = g.sort_values("month")
        ax.plot(g["month"], g["gust_m_s"], linewidth=0.8, label=POINT_LABELS.get(point, point), color=COLORS.get(point))
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle=":", linewidth=1.5, label=f"{threshold:g} m/s")
    ax.set_xlabel("Month")
    ax.set_ylabel("m/s")
    ax.set_title(f"{title_label(label)} - monthly maximum, {month_range_label(start_month, end_month)} only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def seasonal_max(df, start_month, end_month):
    """Max per point/year within the [start_month, end_month] window only
    (default Aug-Nov, the growing-season stem-lodging risk period - see
    README), not the full calendar year. A year with no data in that window
    (e.g. the record's first/last partial year) simply drops out via the
    groupby, no separate completeness check needed."""
    df = df[(df["date"].dt.month >= start_month) & (df["date"].dt.month <= end_month)].copy()
    df["year"] = df["date"].dt.year
    return df.groupby(["point", "year"], as_index=False)["gust_m_s"].max()


def plot_seasonal_max(seasonal, label, start_month, end_month, out_path, markers=True, threshold=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    for point, g in seasonal.groupby("point"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["gust_m_s"], marker="o" if markers else None, markersize=3, linewidth=1, label=POINT_LABELS.get(point, point), color=COLORS.get(point))
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle=":", linewidth=1.5, label=f"{threshold:g} m/s")
    ax.set_xlabel("Year")
    ax.set_ylabel("m/s")
    ax.set_title(f"{title_label(label)} - annual max within {month_range_label(start_month, end_month)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(csv_path, out_dir, variable, start_month, end_month, threshold):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    tag, label = tag_and_label(csv_path, variable)
    season_tag = f"{tag}_m{start_month:02d}-{end_month:02d}"

    monthly = monthly_max(df)
    monthly_csv = out_dir / f"gust_monthly_max_{tag}.csv"
    monthly.to_csv(monthly_csv, index=False)
    print(f"Wrote {monthly_csv}")
    plot_monthly_max(monthly, label, out_dir / f"gust_monthly_max_{tag}.png")

    monthly_season = monthly_max_in_season(monthly, start_month, end_month, connect=False)
    plot_monthly_max_season(monthly_season, label, start_month, end_month, threshold, out_dir / f"gust_monthly_max_{season_tag}.png")

    monthly_season_connected = monthly_max_in_season(monthly, start_month, end_month, connect=True)
    plot_monthly_max_season(monthly_season_connected, label, start_month, end_month, threshold, out_dir / f"gust_monthly_max_{season_tag}_connected.png")

    seasonal = seasonal_max(df, start_month, end_month)
    seasonal_csv = out_dir / f"gust_seasonal_max_{season_tag}.csv"
    seasonal.to_csv(seasonal_csv, index=False)
    print(f"Wrote {seasonal_csv}")
    plot_seasonal_max(seasonal, label, start_month, end_month, out_dir / f"gust_seasonal_max_{season_tag}.png", markers=True, threshold=threshold)
    plot_seasonal_max(seasonal, label, start_month, end_month, out_dir / f"gust_seasonal_max_{season_tag}_nomarkers.png", markers=False, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--variable", default=None, choices=list(VARIABLES), help="Overrides auto-detection from the CSV filename")
    parser.add_argument("--start-month", type=int, default=8, help="Season start month (default Aug)")
    parser.add_argument("--end-month", type=int, default=11, help="Season end month (default Nov)")
    parser.add_argument("--threshold", type=float, default=22.0, help="Dotted horizontal line on the seasonal monthly-max plot, m/s (default 22, the Pinera-Chavez stem-lodging threshold)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.csv), Path(args.out_dir), args.variable, args.start_month, args.end_month, args.threshold)
