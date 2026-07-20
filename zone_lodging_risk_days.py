#!/usr/bin/env python3
"""Count the root-lodging-risk days in the Aug-Dec growing season, per year, for
each GRDC cropping zone and the two Wagga paddock points.

A day is "at risk of root lodging" when the soil is likely still wet: it rained
at least `--threshold` mm, or on any of the previous `--window-days` days (the
wet day plus the days after it). Wetness is judged from each zone's `--rain-stat`
daily rainfall (default the zone mean, matching the gust aggregation) and, for
Wagga, the mean of the two points' rainfall. The wet flag is carried across the
full daily record before
the Aug-Dec days are counted, so a late-July soaking still puts early-August
days at risk.

Writes:
  * lodging_risk_days_per_year_<tag>.csv - one row per year, one column per zone
    (and Wagga): the number of at-risk days in that year's Aug-Dec window.
  * lodging_risk_days_summary_<tag>.csv - per location, the mean / min / max /
    latest at-risk-day count per year, sorted by mean (driest to wettest).

Reads only the rainfall CSVs from extract_rainfall_history.py - no BARRA2 access.

Example:
    python zone_lodging_risk_days.py \\
        --rain-zone-csv /scratch/xe2/cb8590/wind-gusts2/rain_zone_summary_pr_197901-202601.csv \\
        --rain-point-csv /scratch/xe2/cb8590/wind-gusts2/rain_history_pr_197901-202601.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from plot_combined_gust_probabilities import (
    SEASON, at_risk_mask, load_wagga_rain, load_zone_rain,
)

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
WAGGA_LABEL = "Wagga Wagga (2-point mean)"


def rain_tag(rain_zone_csv):
    tag = rain_zone_csv.stem
    for sep in ("_medians_", "_summary_"):
        if sep in tag:
            return tag.split(sep, 1)[1]
    return tag


def season_risk_days_per_year(rain, threshold, window_days):
    """Series indexed by year: count of at-risk days falling in Aug-Dec."""
    mask = at_risk_mask(rain, threshold, window_days)
    sm, em = SEASON
    in_season = mask[mask.index.month.to_series(index=mask.index).between(sm, em)]
    return in_season.groupby(in_season.index.year).sum().astype(int)


def main(rain_zone_csv, rain_point_csv, out_dir, threshold, window_days, rain_stat):
    series = load_zone_rain(rain_zone_csv, rain_stat)  # zone {stat} daily rainfall
    series[WAGGA_LABEL] = load_wagga_rain(rain_point_csv)

    per_year = pd.DataFrame({name: season_risk_days_per_year(s, threshold, window_days)
                             for name, s in series.items()})
    per_year.index.name = "year"
    # Full Aug-Dec seasons only (drop a trailing partial year like 2026).
    per_year = per_year.dropna(how="any")
    per_year = per_year.astype(int)

    tag = rain_tag(rain_zone_csv)
    thr_tag = f"{tag}_rain{threshold:g}mm{window_days}d"
    per_year_csv = out_dir / f"lodging_risk_days_per_year_{thr_tag}.csv"
    per_year.to_csv(per_year_csv)
    print(f"Wrote {per_year_csv} ({len(per_year)} years x {per_year.shape[1]} locations)")

    summary = pd.DataFrame({
        "mean_days_per_year": per_year.mean().round(1),
        "min_days_per_year": per_year.min(),
        "max_days_per_year": per_year.max(),
        "latest_year_days": per_year.iloc[-1],
    }).sort_values("mean_days_per_year")
    summary.index.name = "location"
    summary_csv = out_dir / f"lodging_risk_days_summary_{thr_tag}.csv"
    summary.to_csv(summary_csv)
    print(f"Wrote {summary_csv}\n")
    print(f"At-risk days per year in Aug-Dec (zone {rain_stat} rain >= {threshold:g} mm + "
          f"{window_days} days), {per_year.index.min()}-{per_year.index.max()}:")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rain-zone-csv", required=True)
    parser.add_argument("--rain-point-csv", required=True)
    parser.add_argument("--threshold", type=float, default=10.0, help="mm/day that counts as a wet day")
    parser.add_argument("--window-days", type=int, default=3, help="risk days after each wet day")
    parser.add_argument("--rain-stat", default="mean", choices=["min", "mean", "median", "p75", "p90", "max"],
                        help="per-zone rainfall statistic used to judge wetness (default mean)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.rain_zone_csv), Path(args.rain_point_csv), Path(args.out_dir),
         args.threshold, args.window_days, args.rain_stat)
