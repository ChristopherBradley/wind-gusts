#!/usr/bin/env python3
"""25-year (annual exceedance probability 0.04) return-level gust per GRDC
cropping zone, and - optionally - written back onto the zones vector file as
an attribute.

For each zone we take the Aug-Nov seasonal-maximum of the daily zone-<stat>
gust series for every year, then read off the gust speed with an annual
exceedance probability of 0.04 - the level the annual maximum equals or
exceeds in 4% of years, i.e. the empirical 1-in-25-year gust. This is the
speed at which that zone's curve in gust_zone_exceedance_prob crosses annual
probability 0.04. Empirical (a quantile of the observed annual maxima), to
stay consistent with those empirical curves - no distribution is fitted.

Writes:
  * gust_zone_25yr_return_level_<stat>_<tag>_m08-11.csv - one row per zone.
  * (unless --no-gpkg) adds/updates a real-valued column on --gpkg, keyed by
    the zone name column, holding each zone's 25-year return level in m/s.

Reads the wide daily CSV from extract_zone_gust_history.py - no BARRA2 access.

Example:
    python zone_gust_25yr_return_level.py --stat median \\
        --csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plot_zone_gust_probabilities import detect_variable, seasonal_annual_max

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
GPKG = Path("/home/147/cb8590/Projects/wind-gusts2/grdc_cropping_zones.gpkg")
ZONE_COL = "AEZ"
ANNUAL_EXCEEDANCE_P = 0.04  # 1-in-25-year


def return_level(annual_max_values, p):
    """Empirical gust with annual exceedance probability p: the (1 - p)
    quantile of the annual maxima (linear interpolation between order stats)."""
    return float(np.quantile(annual_max_values, 1.0 - p))


def main(csv_path, out_dir, stat, start_month, end_month, gpkg, attr, write_gpkg):
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    tag = csv_path.stem
    if tag.startswith("gust_zone_summary_"):
        tag = tag[len("gust_zone_summary_"):]

    suffix = f"_{stat}"
    zone_cols = [c for c in df.columns if c.endswith(suffix)]
    if not zone_cols:
        raise SystemExit(f"No columns ending in {suffix!r} in {csv_path}")
    zones = [c[: -len(suffix)] for c in zone_cols]

    rows = []
    for zone, col in zip(zones, zone_cols):
        am = seasonal_annual_max(df[col], start_month, end_month)
        rows.append(dict(zone=zone, n_years=len(am),
                         gust_25yr_ms=round(return_level(am.values, ANNUAL_EXCEEDANCE_P), 1)))
    table = pd.DataFrame(rows)

    season_tag = f"{stat}_{tag}_m{start_month:02d}-{end_month:02d}"
    table_csv = out_dir / f"gust_zone_25yr_return_level_{season_tag}.csv"
    table.to_csv(table_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {table_csv}")

    if write_gpkg:
        import fiona
        import geopandas as gpd

        layer = fiona.listlayers(str(gpkg))[0]
        gdf = gpd.read_file(str(gpkg), layer=layer)
        value_by_zone = dict(zip(table["zone"], table["gust_25yr_ms"]))
        gdf[attr] = gdf[ZONE_COL].map(value_by_zone)
        missing = gdf[gdf[attr].isna()][ZONE_COL].tolist()
        if missing:
            raise SystemExit(f"No return level computed for zones: {missing}")
        # Write to a sibling temp file then atomically replace, so the layer
        # name is preserved and a failure can't leave a half-written gpkg.
        tmp = gpkg.with_name(gpkg.stem + ".tmp.gpkg")
        if tmp.exists():
            tmp.unlink()
        gdf.to_file(str(tmp), layer=layer, driver="GPKG")
        os.replace(tmp, gpkg)
        print(f"Added attribute {attr!r} to {gpkg} (layer {layer!r})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--stat", default="median", choices=["median", "p75", "p90", "max"])
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--gpkg", default=str(GPKG))
    parser.add_argument("--attr", default="gust25yr_median_ms", help="gpkg attribute name to add")
    parser.add_argument("--no-gpkg", dest="write_gpkg", action="store_false", help="Only write the CSV table")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.csv), Path(args.out_dir), args.stat, args.start_month, args.end_month,
         Path(args.gpkg), args.attr, args.write_gpkg)
