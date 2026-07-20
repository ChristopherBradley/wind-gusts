#!/usr/bin/env python3
"""Map of where the "Maximum - Australia" and "Minimum - Australia" envelope
lines (in plot_combined_gust_probabilities.py) come from.

Those envelope lines are the daily windiest and calmest single pixel across all
GRDC cropping zones, so they sit far above/below the zone means. This script
shows *where* those pixels are: for every cropping pixel it computes the mean
Aug-Dec seasonal-maximum gust (wsgsmax * 0.9) over the whole record, and it
tallies how often each pixel is the domain-wide daily maximum or minimum during
Aug-Dec. It then plots the mean-seasonal-max field over the cropping zones and
marks the pixel that is most often the daily windiest (drives the Maximum line)
and most often the daily calmest (drives the Minimum line).

WA Ord and WA Mallee are excluded (no wheat), matching the plots. Reads BARRA2
- run it on a compute node (see the PBS example below).

Example (full history):
    python make_extreme_pixel_map.py
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barra_common import available_month_range, load_zone_labels, open_variable, prepare_spatial

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")
GPKG = Path("/home/147/cb8590/Projects/wind-gusts2/grdc_cropping_zones.gpkg")
LABEL_COL = "AEZ"
EXCLUDE_ZONES = {"WA Ord", "WA Mallee"}
SEASON = (8, 12)  # Aug-Dec


def compute(variable, start_year, end_year):
    sm, em = SEASON
    first = open_variable(variable, start_year, sm, start_year, sm).isel(time=0)
    ref = prepare_spatial(first)
    lats, lons = ref["lat"].values, ref["lon"].values
    zone_id_grid, id_to_name = load_zone_labels(str(GPKG), ref, LABEL_COL)
    exclude_ids = {i for i, n in id_to_name.items() if n in EXCLUDE_ZONES}
    mask = np.isin(zone_id_grid, [i for i in id_to_name if i not in exclude_ids])
    H, W = mask.shape
    midx = np.flatnonzero(mask.ravel())

    sum_smax = np.zeros((H, W), dtype="float64")
    n_years = 0
    max_count = np.zeros(H * W, dtype="int64")
    min_count = np.zeros(H * W, dtype="int64")

    t0 = time.time()
    for year in range(start_year, end_year + 1):
        da = prepare_spatial(open_variable(variable, year, sm, year, em))
        vals = np.asarray(da.values, dtype="float32")  # (T, H, W)
        sum_smax += np.nan_to_num(np.nanmax(vals, axis=0))
        n_years += 1
        flat = vals.reshape(vals.shape[0], -1)[:, midx]  # (T, n_mask)
        np.add.at(max_count, midx[np.nanargmax(flat, axis=1)], 1)
        np.add.at(min_count, midx[np.nanargmin(flat, axis=1)], 1)
        print(f"{year}: {vals.shape[0]} days ({time.time() - t0:.0f}s)", flush=True)

    mean_smax = np.where(mask, sum_smax / n_years, np.nan)
    return dict(mean_smax=mean_smax, mask=mask, lats=lats, lons=lons,
                zone_id_grid=zone_id_grid, id_to_name=id_to_name,
                max_count=max_count, min_count=min_count, W=W, n_years=n_years)


def pixel_info(flat_idx, res):
    row, col = divmod(flat_idx, res["W"])
    lat, lon = float(res["lats"][row]), float(res["lons"][col])
    zone = res["id_to_name"].get(int(res["zone_id_grid"][row, col]), "?")
    return lat, lon, zone, float(res["mean_smax"][row, col])


def plot(res, variable, tag, out_dir):
    import geopandas as gpd

    windiest = int(np.argmax(res["max_count"]))
    calmest = int(np.argmax(res["min_count"]))
    wlat, wlon, wzone, wval = pixel_info(windiest, res)
    clat, clon, czone, cval = pixel_info(calmest, res)

    fig, ax = plt.subplots(figsize=(11, 10))
    lons, lats = res["lons"], res["lats"]
    mesh = ax.pcolormesh(lons, lats, res["mean_smax"], cmap="viridis", shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Aug–Dec seasonal-maximum gust (m/s)")

    gdf = gpd.read_file(str(GPKG))
    gdf.boundary.plot(ax=ax, color="0.25", linewidth=0.4, zorder=3)

    ax.plot(wlon, wlat, marker="*", markersize=22, color="red", markeredgecolor="k", zorder=5,
            label=f"Windiest pixel → Maximum-Australia\n{wzone} ({wlat:.2f}, {wlon:.2f}), mean {wval:.1f} m/s")
    ax.plot(clon, clat, marker="*", markersize=22, color="deepskyblue", markeredgecolor="k", zorder=5,
            label=f"Calmest pixel → Minimum-Australia\n{czone} ({clat:.2f}, {clon:.2f}), mean {cval:.1f} m/s")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title(f"Where the Australia gust envelope comes from ({variable.replace('_', ' ')})\n"
                 f"cropping pixels, most-frequent daily Aug–Dec max/min over {res['n_years']} years")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    out_path = out_dir / f"gust_extreme_pixel_map_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWindiest (Maximum-Australia): {wzone} ({wlat:.3f}, {wlon:.3f}), "
          f"daily-max on {int(res['max_count'][windiest])} days, mean seasonal max {wval:.1f} m/s")
    print(f"Calmest (Minimum-Australia): {czone} ({clat:.3f}, {clon:.3f}), "
          f"daily-min on {int(res['min_count'][calmest])} days, mean seasonal max {cval:.1f} m/s")
    print(f"Wrote {out_path}")


def main(variable, start_year, end_year, out_dir):
    res = compute(variable, start_year, end_year)
    tag = f"{variable}_{start_year}-{end_year}_m{SEASON[0]:02d}-{SEASON[1]:02d}"
    plot(res, variable, tag, out_dir)


if __name__ == "__main__":
    auto_sy, _, auto_ey, auto_em = available_month_range("wsgsmax")
    end_year = auto_ey if auto_em >= SEASON[1] else auto_ey - 1  # need a full Aug-Dec
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", default="wsgsmax_bias")
    parser.add_argument("--start-year", type=int, default=auto_sy)
    parser.add_argument("--end-year", type=int, default=end_year)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.variable, args.start_year, args.end_year, Path(args.out_dir))
