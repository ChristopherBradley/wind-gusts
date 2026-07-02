#!/usr/bin/env python3
"""Extract a daily time series of a BARRA2 daily-max variable at a given
lat/lon point, saving both a CSV and a plot.

Coordinates are given as (lat, lon) in degrees, e.g. for Mt Kosciuszko:
    python extract_point_timeseries.py --variable tasmax --lat -36.456671 --lon 148.262724 \
        --start-year 2022 --start-month 9 --end-year 2022 --end-month 11
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from barra_common import VARIABLES, open_variable, prepare_spatial

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def extract_point_timeseries(
    variable, lat, lon, start_year, start_month, end_year, end_month, out_dir=OUT_DIR
):
    cfg = VARIABLES[variable]
    da = open_variable(variable, start_year, start_month, end_year, end_month)
    da = prepare_spatial(da)

    point = da.sel(lat=lat, lon=lon, method="nearest").compute()
    pixel_lat = float(point["lat"])
    pixel_lon = float(point["lon"])
    print(f"Requested ({lat}, {lon}) -> nearest grid cell ({pixel_lat:.4f}, {pixel_lon:.4f})")

    col_name = f"{variable}_{cfg['units_out']}"
    series = point.to_series()
    series.index.name = "date"
    series.name = col_name

    df = series.reset_index()
    df["date"] = df["date"].dt.date

    tag = f"{variable}_{start_year}{start_month:02d}-{end_year}{end_month:02d}"
    csv_path = out_dir / f"{tag}_timeseries.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df[col_name], marker="o", markersize=3)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{cfg['long_name']} ({cfg['units_out']})")
    ax.set_title(f"{cfg['long_name']} at ({pixel_lat:.4f}, {pixel_lon:.4f})")
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_path = out_dir / f"{tag}_timeseries.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    return df, csv_path, plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="tasmax", choices=list(VARIABLES))
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--end-month", type=int, default=11)
    args = parser.parse_args()
    extract_point_timeseries(
        args.variable,
        args.lat,
        args.lon,
        args.start_year,
        args.start_month,
        args.end_year,
        args.end_month,
    )
