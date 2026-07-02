#!/usr/bin/env python3
"""Build an Australia-wide GeoTIFF of the maximum of a BARRA2 daily-max
variable over a month (or a longer year/month range) - e.g. the hottest
day's tasmax during September, or the windiest day's wsgsmax gust across an
entire season.

Saved as uint8 with an embedded GDAL colour table so it auto-colours in
QGIS. The colour scale defaults to the actual min/max of the output data.

Example:
    python make_monthly_max_geotiff.py --variable tasmax --year 2022 --month 9
    python make_monthly_max_geotiff.py --variable wsgsmax --year 2022 --month 8 --end-month 11
"""
import argparse
from pathlib import Path

from barra_common import (
    VARIABLES,
    open_variable,
    prepare_spatial,
    to_uint8_with_colormap,
    write_geotiff_with_colormap,
)

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def make_monthly_max_geotiff(
    variable, year, month, end_year=None, end_month=None, vmin=None, vmax=None, cmap=None, out_path=None
):
    cfg = VARIABLES[variable]
    cmap = cfg["cmap"] if cmap is None else cmap

    da = open_variable(variable, year, month, end_year, end_month)
    da_max = da.max(dim="time", keep_attrs=True).compute()
    da_max = prepare_spatial(da_max)

    uint8_da, colormap, used_vmin, used_vmax = to_uint8_with_colormap(da_max, vmin, vmax, cmap)

    if out_path is None:
        tag = f"{year}{month:02d}"
        if end_year is not None:
            tag += f"-{end_year}{end_month:02d}"
        out_path = OUT_DIR / f"{variable}_max_{tag}.tif"
    write_geotiff_with_colormap(uint8_da, colormap, out_path)
    print(f"variable={variable} value_range=[{used_vmin:.3f}, {used_vmax:.3f}] cmap={cmap}")
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="tasmax", choices=list(VARIABLES))
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--end-year", type=int, default=None, help="End of an optional multi-month range")
    parser.add_argument("--end-month", type=int, default=None, help="End of an optional multi-month range")
    parser.add_argument("--vmin", type=float, default=None, help="Override the auto (data min) colour-scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Override the auto (data max) colour-scale maximum")
    parser.add_argument("--cmap", default=None, help="Override the default matplotlib colormap name")
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args()
    make_monthly_max_geotiff(
        args.variable, args.year, args.month, args.end_year, args.end_month, args.vmin, args.vmax, args.cmap, args.out_path
    )
