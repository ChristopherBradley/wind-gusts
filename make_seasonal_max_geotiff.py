#!/usr/bin/env python3
"""Build an Australia-wide GeoTIFF of the maximum of a BARRA2 daily-max
variable seen across the same month range repeated over multiple years -
e.g. the windiest Aug-Nov day anywhere in 2017-2025's wsgsmax gust.

Whole region, no land-use/zone masking (compare landuse_wind_year.py for
the masked equivalent). Processes one year at a time and folds it into a
running elementwise max so memory stays bounded to a single season's data
regardless of how many years are requested.

Saved as uint8 with an embedded GDAL colour table, same as
make_monthly_max_geotiff.py.

Example:
    python make_seasonal_max_geotiff.py --variable wsgsmax --start-year 2017 --end-year 2025 --start-month 8 --end-month 11
"""
import argparse
import time
from pathlib import Path

import numpy as np

from barra_common import (
    VARIABLES,
    open_variable,
    prepare_spatial,
    to_uint8_direct,
    to_uint8_with_colormap,
    write_geotiff_with_colormap,
)

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def make_seasonal_max_geotiff(
    variable,
    start_year,
    end_year,
    start_month,
    end_month,
    vmin=None,
    vmax=None,
    cmap=None,
    out_path=None,
    direct_units=False,
):
    t0 = time.time()
    cfg = VARIABLES[variable]
    cmap = cfg["cmap"] if cmap is None else cmap

    template = None
    running_max = None
    for year in range(start_year, end_year + 1):
        da = open_variable(variable, year, start_month, year, end_month)
        da_max = da.max(dim="time", keep_attrs=True).compute()
        if template is None:
            template = da_max
            running_max = da_max.values.copy()
        else:
            running_max = np.fmax(running_max, da_max.values)
        print(f"{year}: done ({time.time() - t0:.1f}s)")

    overall_max = template.copy(data=running_max)
    overall_max = prepare_spatial(overall_max)

    if direct_units:
        uint8_da, colormap, used_vmin, used_vmax = to_uint8_direct(overall_max, cmap)
    else:
        uint8_da, colormap, used_vmin, used_vmax = to_uint8_with_colormap(overall_max, vmin, vmax, cmap)

    if out_path is None:
        tag = f"{start_year}-{end_year}_m{start_month:02d}-{end_month:02d}"
        out_path = OUT_DIR / f"{variable}_seasonal_max_{tag}.tif"
    write_geotiff_with_colormap(uint8_da, colormap, out_path)
    print(f"variable={variable} value_range=[{used_vmin:.3f}, {used_vmax:.3f}] cmap={cmap}")
    print(f"Wrote {out_path} ({time.time() - t0:.1f}s total)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="wsgsmax", choices=list(VARIABLES))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--vmin", type=float, default=None, help="Override the auto (data min) colour-scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Override the auto (data max) colour-scale maximum")
    parser.add_argument("--cmap", default=None, help="Override the default matplotlib colormap name")
    parser.add_argument("--out-path", default=None)
    parser.add_argument(
        "--direct-units",
        action="store_true",
        help="Pixel value N means N native units (e.g. N m/s) instead of being rescaled to 1-255 - "
        "only sensible for variables that stay within ~1-255 in their native units",
    )
    args = parser.parse_args()
    make_seasonal_max_geotiff(
        args.variable,
        args.start_year,
        args.end_year,
        args.start_month,
        args.end_month,
        args.vmin,
        args.vmax,
        args.cmap,
        args.out_path,
        args.direct_units,
    )
