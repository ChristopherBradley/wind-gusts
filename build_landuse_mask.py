#!/usr/bin/env python3
"""Resample an NLUM probSurf land-use raster onto the BARRA grid, threshold
it, intersect with a vector file of zones (e.g. GRDC cropping zones), and
cache the resulting boolean mask as a small uint8 GeoTIFF (1=match, 0=no
match) so downstream per-year scripts don't have to repeat the expensive
reprojection of the (very high resolution) NLUM raster.

Example:
    python build_landuse_mask.py --landuse-tif NLUM_v7_probSurf_2021_331_5_W_CER.tif \
        --threshold 5000 --zones-gpkg grdc_cropping_zones.gpkg
"""
import argparse
import time
from pathlib import Path

from barra_common import load_landuse_mask, load_zone_mask, open_variable, prepare_spatial

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
LANDUSE_TIF = Path("/home/147/cb8590/Projects/wind-gusts2/NLUM_v7_probSurf_2021_331_5_W_CER.tif")
ZONES_GPKG = Path("/home/147/cb8590/Projects/wind-gusts2/grdc_cropping_zones.gpkg")


def main(variable, year, month, landuse_tif, threshold, zones_gpkg, out_path):
    t0 = time.time()
    reference = open_variable(variable, year, month).isel(time=0).compute()
    reference = prepare_spatial(reference)

    landuse_mask = load_landuse_mask(landuse_tif, reference, threshold)
    zone_mask = load_zone_mask(zones_gpkg, reference)
    mask = landuse_mask & zone_mask

    print(
        f"Land-use >{threshold:g}: {int(landuse_mask.sum())} pixels; "
        f"within a cropping zone: {int(zone_mask.sum())} pixels; "
        f"combined: {int(mask.sum())} of {mask.size} BARRA pixels (took {time.time() - t0:.1f}s)"
    )

    mask_uint8 = mask.astype("uint8")
    mask_uint8 = mask_uint8.rio.write_nodata(255)
    mask_uint8.rio.to_raster(out_path, dtype="uint8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="sfcWindmax", help="Any BARRA variable, just used to define the grid")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--landuse-tif", default=str(LANDUSE_TIF))
    parser.add_argument("--threshold", type=float, default=5000.0)
    parser.add_argument("--zones-gpkg", default=str(ZONES_GPKG))
    parser.add_argument("--out-path", default=str(OUT_DIR / "landuse_mask.tif"))
    args = parser.parse_args()
    main(args.variable, args.year, args.month, args.landuse_tif, args.threshold, args.zones_gpkg, args.out_path)
