#!/usr/bin/env python3
"""Daily 5-figure summary of a BARRA-C2 gust variable within each GRDC cropping
zone, for the full history on disk (nominally 1979 onward).

For every day and every polygon in the GRDC cropping-zones vector file, we
summarise all BARRA2 pixels whose centre falls inside that polygon with four
statistics: median, upper quartile (p75), p90 and maximum. With 18 zones that
is 72 value columns plus the date, one row per day.

`--variable` selects which barra_common.VARIABLES entry to extract - default
`wsgsmax_bias` ("Corrected gust": wsgsmax * 0.9, still at 10 m), the same
variable as gust_history_wsgsmax_bias_197901-202601.csv, so this is the
zone-aggregated companion to extract_gust_history.py's two-point CSV.

Two CSVs are written:
  * gust_zone_summary_<tag>.csv   - date + 72 stat columns (all four stats,
    grouped per zone: "<zone>_median", "<zone>_p75", "<zone>_p90",
    "<zone>_max").
  * gust_zone_medians_<tag>.csv   - date + one median column per zone, the
    convenience subset used by the exceedance-probability plots.

Processes one calendar month (one BARRA2 file) at a time to keep the full
spatial grid's memory footprint bounded - this reads the whole Australia
domain, not a handful of points, so it is much heavier than
extract_gust_history.py and is meant for the PBS job.

Example (full history, default variable - the expensive one, for PBS):
    python extract_zone_gust_history.py --variable wsgsmax_bias

Small dry run (fast, for testing the pipeline end-to-end):
    python extract_zone_gust_history.py --start-year 2022 --start-month 9 \\
        --end-year 2022 --end-month 9
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from barra_common import VARIABLES, available_month_range, open_variable, prepare_spatial, load_zone_labels

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
GPKG = Path("/home/147/cb8590/Projects/wind-gusts2/grdc_cropping_zones.gpkg")
LABEL_COL = "AEZ"

# The four-statistic summary. p50 is the median; p75 is the "upper quartile".
PERCENTILES = [50, 75, 90]


def month_stats(values, zone_masks):
    """values: (time, lat, lon) float array for one month, oriented to match
    the zone grid. zone_masks: {name: 2D bool mask}. Returns {column: array}
    with one array of length `time` per (zone, statistic)."""
    out = {}
    for name, mask in zone_masks.items():
        sub = values[:, mask]  # (time, n_pixels_in_zone)
        q50, q75, q90 = np.nanpercentile(sub, PERCENTILES, axis=1)
        out[f"{name}_median"] = q50
        out[f"{name}_p75"] = q75
        out[f"{name}_p90"] = q90
        out[f"{name}_max"] = np.nanmax(sub, axis=1)
    return out


def main(variable, start_year, start_month, end_year, end_month, out_dir):
    t0 = time.time()

    # Build the zone-id grid once from the first month's spatial template - the
    # BARRA2 grid is identical across every file, so this never changes.
    first = open_variable(variable, start_year, start_month, start_year, start_month).isel(time=0)
    ref = prepare_spatial(first)
    zone_id_grid, id_to_name = load_zone_labels(str(GPKG), ref, LABEL_COL)
    # Preserve polygon order (zone id 1..N) so columns come out zone-grouped.
    zone_masks = {id_to_name[z]: (zone_id_grid == z) for z in sorted(id_to_name)}
    for name, mask in zone_masks.items():
        print(f"  zone {name!r}: {int(mask.sum())} pixels")

    frames = []
    for year in range(start_year, end_year + 1):
        m0 = start_month if year == start_year else 1
        m1 = end_month if year == end_year else 12
        for month in range(m0, m1 + 1):
            t1 = time.time()
            da = prepare_spatial(open_variable(variable, year, month, year, month))
            values = np.asarray(da.values, dtype="float32")
            dates = pd.to_datetime(da["time"].values).normalize()
            data = month_stats(values, zone_masks)
            frames.append(pd.DataFrame(data, index=dates))
            print(f"{year}-{month:02d}: {len(dates)} days in {time.time() - t1:.1f}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    out = pd.concat(frames).sort_index()
    out.index.name = "date"

    tag = f"{variable}_{start_year}{start_month:02d}-{end_year}{end_month:02d}"

    summary_csv = out_dir / f"gust_zone_summary_{tag}.csv"
    out.to_csv(summary_csv)
    print(f"\nWrote {summary_csv} ({len(out)} rows x {out.shape[1]} cols)")

    median_cols = [c for c in out.columns if c.endswith("_median")]
    medians = out[median_cols].rename(columns=lambda c: c[: -len("_median")])
    medians_csv = out_dir / f"gust_zone_medians_{tag}.csv"
    medians.to_csv(medians_csv)
    print(f"Wrote {medians_csv} ({len(medians)} rows x {medians.shape[1]} cols)")
    print(f"Total {time.time() - t0:.0f}s")
    return summary_csv, medians_csv


if __name__ == "__main__":
    auto_sy, auto_sm, auto_ey, auto_em = available_month_range("wsgsmax")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="wsgsmax_bias", choices=list(VARIABLES))
    parser.add_argument("--start-year", type=int, default=auto_sy)
    parser.add_argument("--start-month", type=int, default=auto_sm)
    parser.add_argument("--end-year", type=int, default=auto_ey)
    parser.add_argument("--end-month", type=int, default=auto_em)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.variable, args.start_year, args.start_month, args.end_year, args.end_month, Path(args.out_dir))
