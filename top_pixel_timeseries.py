#!/usr/bin/env python3
"""For each year, find the day (in the day-aggregated exceedance table) with
the most pixels exceeding the wind threshold, take its representative
coordinate (south/east corner - see combine_exceedances.py), then plot that
single pixel's full continuous daily-max-wind-speed time series across the
entire 2017-2025 period (all months, not just Sep-Nov).

A year with zero exceedances has no candidate coordinate and is skipped.

Processes one calendar year of data at a time (12 monthly files), selecting
all anchor points at once per year, to keep each step well inside this
node's per-process CPU budget.

Example:
    python top_pixel_timeseries.py --by-day-csv wind_exceedances_2017-2025_by_day.csv
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from barra_common import VARIABLES, open_variable

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def pick_anchors(by_day, start_year, end_year):
    """One (year, date, lat, lon, n_pixels) per year with the most pixels that
    day, ties broken by the higher wind speed. Years with no exceedances are
    dropped (and reported) rather than guessed."""
    anchors = []
    for year in range(start_year, end_year + 1):
        g = by_day[by_day["year"] == year]
        if g.empty:
            print(f"{year}: no exceedances at all - skipping, no anchor coordinate exists")
            continue
        top = g.sort_values(["n_pixels", "max_wind_speed"], ascending=False).iloc[0]
        anchors.append(dict(year=year, date=top["date"], lat=float(top["lat"]), lon=float(top["lon"]), n_pixels=int(top["n_pixels"])))
    return anchors


def extract_full_series(anchors, variable, start_year, end_year):
    lats = xr.DataArray([a["lat"] for a in anchors], dims="point")
    lons = xr.DataArray([a["lon"] for a in anchors], dims="point")

    yearly_frames = []
    for year in range(start_year, end_year + 1):
        da = open_variable(variable, year, 1, year, 12)
        pts = da.sel(lat=lats, lon=lons, method="nearest").compute()
        df = pts.to_pandas()
        df.index.name = "date"
        yearly_frames.append(df)
        print(f"{year}: extracted {df.shape[0]} days for {df.shape[1]} anchor points")

    full = pd.concat(yearly_frames, axis=0)
    full.columns = [f"{a['year']}_anchor" for a in anchors]
    return full


def plot_anchor(series, anchor, variable, out_dir):
    cfg = VARIABLES[variable]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{cfg['long_name']} ({cfg['units_out']})")
    ax.set_title(
        f"Pixel ({anchor['lat']:.2f}, {anchor['lon']:.2f}) - top exceedance day for {anchor['year']} "
        f"({anchor['date']}, n_pixels={anchor['n_pixels']})"
    )
    fig.tight_layout()

    out_path = out_dir / f"top_pixel_timeseries_{anchor['year']}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(by_day_csv, variable, start_year, end_year, out_dir):
    by_day = pd.read_csv(by_day_csv)
    anchors = pick_anchors(by_day, start_year, end_year)

    full = extract_full_series(anchors, variable, start_year, end_year)

    for anchor, col in zip(anchors, full.columns):
        plot_anchor(full[col], anchor, variable, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-day-csv", default=str(OUT_DIR / "wind_exceedances_2017-2025_by_day.csv"))
    parser.add_argument("--variable", default="sfcWindmax", choices=list(VARIABLES))
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.by_day_csv), args.variable, args.start_year, args.end_year, Path(args.out_dir))
