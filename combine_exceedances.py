#!/usr/bin/env python3
"""Combine the per-year CSVs written by wind_exceedances_year.py into one CSV
and one GeoPackage of point locations (EPSG:4326), then build a second,
day-aggregated CSV: pixels exceeding the threshold on the same date are
collapsed into a single row, using the south-most latitude and east-most
longitude seen that day as the representative coordinate, the day's single
highest gust (and that observation's direction/region) as the
representative attributes, and a count of how many pixels were aggregated.

Example:
    python combine_exceedances.py --start-year 2017 --end-year 2025
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def combine(start_year, end_year, out_dir):
    frames = [pd.read_csv(out_dir / f"wind_exceedances_{y}.csv") for y in range(start_year, end_year + 1)]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    return df


def to_gpkg(df, out_path):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf.to_file(out_path, driver="GPKG")


def aggregate_by_day(df):
    rows = []
    for date, group in df.groupby("date", sort=True):
        top = group.loc[group["max_gust"].idxmax()]
        rows.append(
            {
                "date": date,
                "year": top["year"],
                "lat": group["lat"].min(),  # south-most
                "lon": group["lon"].max(),  # east-most
                "max_gust": top["max_gust"],
                "wind_direction_deg": top["wind_direction_deg"],
                "wind_direction_compass": top["wind_direction_compass"],
                "cropping_region": top["cropping_region"],
                "n_pixels": len(group),
            }
        )
    return pd.DataFrame(rows)


def main(start_year, end_year, out_dir):
    df = combine(start_year, end_year, out_dir)
    csv_path = out_dir / f"wind_exceedances_{start_year}-{end_year}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    gpkg_path = out_dir / f"wind_exceedances_{start_year}-{end_year}.gpkg"
    to_gpkg(df, gpkg_path)
    print(f"Wrote {gpkg_path}")

    by_day = aggregate_by_day(df)
    by_day_path = out_dir / f"wind_exceedances_{start_year}-{end_year}_by_day.csv"
    by_day.to_csv(by_day_path, index=False)
    print(f"Wrote {by_day_path} ({len(by_day)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.start_year, args.end_year, Path(args.out_dir))
