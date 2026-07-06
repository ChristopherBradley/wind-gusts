#!/usr/bin/env python3
"""Full-history daily gust time series at the two paddock points near Wagga
Wagga used throughout this project, spanning every month of BARRA-C2 data
available on disk (nominally 1979 onward).

`--variable` selects which barra_common.VARIABLES entry to extract - default
`wsgsmax_bias` ("Corrected gust": wsgsmax * 0.9, still at 10 m). Also useful:
`wsgsmax` (bias- AND height-adjusted to 2 m crop height, the main pipeline's
gust and the one directly comparable to the 18-22 m/s Pinera-Chavez lodging
thresholds).

Just the raw daily values, no month/season filtering, no rainfall, no
sfcWindmax - this is the source-of-truth CSV that the monthly-max plot
(plot_gust_climatology.py) and the return-period table
(gust_return_period_table.py) both build from, so re-plotting or re-fitting
never has to touch the BARRA2 files again.

Processes one calendar year (12 monthly files) at a time to stay within the
compute node's per-process memory/CPU budget - see barra_common.open_variable.

Example (full history, both paddock points - this is the expensive one,
meant for the PBS job):
    python extract_gust_history.py --variable wsgsmax_bias
    python extract_gust_history.py --variable wsgsmax

Small dry run (fast, for testing the pipeline end-to-end):
    python extract_gust_history.py --start-year 2023 --end-year 2023
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from barra_common import VARIABLES, available_month_range, open_variable

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

# The two paddock points near Wagga Wagga, NSW, used throughout this project
# (see README.md step 5 / compare_point_gusts.py).
POINTS = [
    dict(name="paddock_2024", lat=-35.050418, lon=147.318795),
    dict(name="paddock_2025", lat=-35.025873, lon=147.354694),
]


def extract_year(variable, year, start_month, end_month, points):
    """Vectorized nearest-neighbour selection of all points at once, so each
    monthly file is only read once regardless of how many points there are."""
    lats = xr.DataArray([p["lat"] for p in points], dims="point", coords={"point": [p["name"] for p in points]})
    lons = xr.DataArray([p["lon"] for p in points], dims="point", coords={"point": [p["name"] for p in points]})
    da = open_variable(variable, year, start_month, year, end_month)
    pt = da.sel(lat=lats, lon=lons, method="nearest").compute()
    return pt.to_dataframe(name="gust_m_s").reset_index()


def main(variable, start_year, start_month, end_year, end_month, out_dir, points):
    frames = []
    t0 = time.time()
    for year in range(start_year, end_year + 1):
        m0 = start_month if year == start_year else 1
        m1 = end_month if year == end_year else 12
        t1 = time.time()
        frames.append(extract_year(variable, year, m0, m1, points))
        print(f"{year} ({m0:02d}-{m1:02d}): {len(frames[-1])} point-days in {time.time() - t1:.1f}s", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"time": "date", "lat": "pixel_lat", "lon": "pixel_lon"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    requested = {p["name"]: (p["lat"], p["lon"]) for p in points}
    out["requested_lat"] = out["point"].map(lambda n: requested[n][0])
    out["requested_lon"] = out["point"].map(lambda n: requested[n][1])
    out = out[["date", "point", "requested_lat", "requested_lon", "pixel_lat", "pixel_lon", "gust_m_s"]]
    out = out.sort_values(["point", "date"]).reset_index(drop=True)

    tag = f"{variable}_{start_year}{start_month:02d}-{end_year}{end_month:02d}"
    csv_path = out_dir / f"gust_history_{tag}.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(out)} rows) in {time.time() - t0:.1f}s total")
    return csv_path


if __name__ == "__main__":
    auto_start_y, auto_start_m, auto_end_y, auto_end_m = available_month_range("wsgsmax")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="wsgsmax_bias", choices=list(VARIABLES))
    parser.add_argument("--start-year", type=int, default=auto_start_y)
    parser.add_argument("--start-month", type=int, default=auto_start_m)
    parser.add_argument("--end-year", type=int, default=auto_end_y)
    parser.add_argument("--end-month", type=int, default=auto_end_m)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.variable, args.start_year, args.start_month, args.end_year, args.end_month, Path(args.out_dir), POINTS)
