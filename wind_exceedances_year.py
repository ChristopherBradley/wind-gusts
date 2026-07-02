#!/usr/bin/env python3
"""For a single year, find every (pixel, date) in Aug-Nov where the adjusted
wsgsmax wind gust exceeds a threshold within the cropping mask (see
build_landuse_mask.py), and record date, year, coordinate, max gust, wind
direction (from the daily-mean uas/vas - an approximation, not tied to the
specific hour of the day's peak gust), and GRDC cropping region. Writes one
CSV per year; combine them with combine_exceedances.py.

The gust (wsgsmax) is bias- and height-adjusted to a 2 m crop-height gust in
barra_common.py, so it is directly comparable to the 22 m/s Pinera-Chavez
(2016) stem-lodging threshold.

Split per-year (rather than looping years in one process) for the same
reason as landuse_wind_year.py: this node has a hard 1800 CPU-second budget
per process on a single CPU.

Example:
    python wind_exceedances_year.py --year 2022 --threshold 22
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401

from barra_common import load_zone_labels, open_variable, prepare_spatial, wind_uv_to_direction

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
MASK_PATH = OUT_DIR / "landuse_mask.tif"
ZONES_GPKG = Path("/home/147/cb8590/Projects/wind-gusts2/grdc_cropping_zones.gpkg")


def load_mask(mask_path):
    """Plain boolean numpy array - see landuse_wind_year.py's load_mask for
    why this must not be combined with netCDF-derived DataArrays as labelled
    xarray objects (dims/coords don't round-trip identically through a tif)."""
    mask_da = rioxarray.open_rasterio(mask_path, masked=False).squeeze("band", drop=True)
    return mask_da.values == 1


def main(year, start_month, end_month, threshold, mask_path, zones_gpkg, out_dir):
    t0 = time.time()
    mask = load_mask(mask_path)

    gust = prepare_spatial(open_variable("wsgsmax", year, start_month, year, end_month))
    u = prepare_spatial(open_variable("uas", year, start_month, year, end_month))
    v = prepare_spatial(open_variable("vas", year, start_month, year, end_month))

    zone_id_grid, id_to_name = load_zone_labels(zones_gpkg, gust.isel(time=0), "AEZ")

    lat_vals = gust["lat"].values
    lon_vals = gust["lon"].values
    time_vals = gust["time"].values
    gust_vals = gust.values
    u_vals = u.values
    v_vals = v.values

    exceed = mask[None, :, :] & (gust_vals > threshold)
    t_idx, r_idx, c_idx = np.nonzero(exceed)

    deg, compass = wind_uv_to_direction(u_vals[t_idx, r_idx, c_idx], v_vals[t_idx, r_idx, c_idx])
    zone_ids = zone_id_grid[r_idx, c_idx]

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(time_vals[t]).date() for t in t_idx],
            "year": year,
            "lat": lat_vals[r_idx],
            "lon": lon_vals[c_idx],
            "max_gust": gust_vals[t_idx, r_idx, c_idx],
            "wind_direction_deg": deg,
            "wind_direction_compass": compass,
            "cropping_region": [id_to_name.get(int(z), "") for z in zone_ids],
        }
    )
    df = df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)

    out_path = out_dir / f"wind_exceedances_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"{year}: {len(df)} exceedances ({time.time() - t0:.1f}s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=22.0)
    parser.add_argument("--mask-path", default=str(MASK_PATH))
    parser.add_argument("--zones-gpkg", default=str(ZONES_GPKG))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.year, args.start_month, args.end_month, args.threshold, args.mask_path, args.zones_gpkg, Path(args.out_dir))
