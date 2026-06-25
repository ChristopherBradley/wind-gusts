#!/usr/bin/env python3
"""Process a single year: take the max of a BARRA2 daily-max variable over a
month range, restrict to pixels matching a cached land-use mask (see
build_landuse_mask.py), write a uint8 GeoTIFF, and save the masked pixel
values to a .npy for later box-plotting (see landuse_wind_boxplot.py).

Split out per-year (rather than looping years in one process) because each
process on this node gets a hard 1800 CPU-second budget on a single CPU -
one year at a time comfortably fits, a 9-year loop in one process did not.

Example:
    python landuse_wind_year.py --variable sfcWindmax --year 2022 --start-month 9 --end-month 11
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401

from barra_common import VARIABLES, open_variable, prepare_spatial, to_uint8_with_colormap, write_geotiff_with_colormap

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
MASK_PATH = OUT_DIR / "landuse_mask.tif"


def load_mask(mask_path):
    """Return the cached mask as a plain boolean numpy array. Deliberately not
    an xarray object: the mask is reloaded from a GeoTIFF (dims y/x, coords
    regenerated from the affine transform) while the BARRA grid uses dims
    lat/lon with coords straight from the netCDF - close but not bit-identical.
    Combining them as labelled DataArrays makes xarray treat y/x and lat/lon
    as unrelated dimensions and broadcast a 1018x1298x1018x1298 array instead
    of aligning elementwise. Comparing the same shape and dimension order
    positionally as numpy arrays avoids that entirely."""
    mask_da = rioxarray.open_rasterio(mask_path, masked=False).squeeze("band", drop=True)
    return mask_da.values == 1


def main(variable, year, start_month, end_month, mask_path, out_dir):
    t0 = time.time()
    cfg = VARIABLES[variable]
    mask = load_mask(mask_path)

    da = open_variable(variable, year, start_month, year, end_month)
    da_max = da.max(dim="time", keep_attrs=True).compute()
    da_max = prepare_spatial(da_max)

    masked = da_max.where(mask)
    masked = prepare_spatial(masked)  # .where() silently drops the CRS/spatial_ref coord - reattach it
    arr = masked.values
    valid = arr[np.isfinite(arr)]

    uint8_da, colormap, vmin, vmax = to_uint8_with_colormap(masked, cmap_name=cfg["cmap"])
    tif_path = out_dir / f"{variable}_landuse_max_{year}{start_month:02d}-{year}{end_month:02d}.tif"
    write_geotiff_with_colormap(uint8_da, colormap, tif_path)

    npy_path = out_dir / f"{variable}_landuse_values_{year}.npy"
    np.save(npy_path, valid)

    print(f"{year}: n={valid.size} range=[{vmin:.2f}, {vmax:.2f}] ({time.time() - t0:.1f}s)")
    print(f"Wrote {tif_path}")
    print(f"Wrote {npy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="sfcWindmax", choices=list(VARIABLES))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=9)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--mask-path", default=str(MASK_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.variable, args.year, args.start_month, args.end_month, args.mask_path, Path(args.out_dir))
