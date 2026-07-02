#!/usr/bin/env python3
"""Overlay the different daily wind/gust estimates at a single lat/lon point for
one season, so the model gust (wsgsmax) and the Pinera-Chavez empirical gust
(reconstructed from sfcWindmax) can be compared side by side. Plots five daily
series over the season (Aug-Nov by default):

    Raw maximum        - sfcWindmax: daily-max hourly-MEAN wind at 10 m (as-is)
    Corrected maximum  - that mean brought to 2 m crop height (x0.575)
    Raw gust           - wsgsmax: the model gust at 10 m (as-is)
    Corrected gust     - wsgsmax bias- & height-adjusted to a 2 m gust (x0.518)
    Pinera-Chavez gust - a 2 m gust reconstructed from the mean via Berry/Pinera
                         Eq. 6 (mean x0.575 x gust-factor ~2.97)

Every factor lives in barra_common.VARIABLES (single source of truth); this
script only reads the two on-disk variables (sfcWindmax, wsgsmax) once each and
applies each logical variable's `convert` to the extracted point series. Writes
one PNG (all five lines + the 22 m/s stem-lodging threshold) and one CSV (one
column per estimate).

Example (the two paddock point-years near Wagga Wagga):
    python compare_point_gusts.py --lat -35.050418 --lon 147.318795 --year 2024
    python compare_point_gusts.py --lat -35.025873 --lon 147.354694 --year 2025
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from barra_common import VARIABLES, open_variable, prepare_spatial

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

# (logical variable, legend label, matplotlib line kwargs). Mean-wind quantities
# are blue, model-gust quantities orange, the Pinera reconstruction green;
# dashed = as-is at 10 m, solid = corrected to 2 m.
SERIES = [
    ("sfcWindmax", "Raw maximum (10 m mean)", dict(color="tab:blue", linestyle="--")),
    ("sfcWindmax_2m", "Corrected maximum (2 m mean)", dict(color="tab:blue", linestyle="-")),
    ("wsgsmax_raw", "Raw gust (10 m)", dict(color="tab:orange", linestyle="--")),
    ("wsgsmax", "Corrected gust (2 m)", dict(color="tab:orange", linestyle="-")),
    ("pinera_gust", "Pinera-Chavez gust (2 m)", dict(color="tab:green", linestyle="-")),
]


def point_series(lat, lon, year, start_month, end_month):
    """Return (dataframe indexed by date with one column per SERIES label,
    nearest-pixel lat, nearest-pixel lon). Reads each on-disk source once."""
    # The identity-convert variables sfcWindmax / wsgsmax_raw give the raw source
    # fields; key them by their on-disk source name so derived series can reuse them.
    raw_by_source = {}
    for src_var in ("sfcWindmax", "wsgsmax_raw"):
        da = prepare_spatial(open_variable(src_var, year, start_month, year, end_month))
        pt = da.sel(lat=lat, lon=lon, method="nearest").compute()
        raw_by_source[VARIABLES[src_var].get("source", src_var)] = pt

    any_pt = next(iter(raw_by_source.values()))
    pixel_lat, pixel_lon = float(any_pt["lat"]), float(any_pt["lon"])

    cols = {}
    for var, label, _ in SERIES:
        cfg = VARIABLES[var]
        src = cfg.get("source", var)
        cols[label] = cfg["convert"](raw_by_source[src]).to_series()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df, pixel_lat, pixel_lon


def main(lat, lon, year, start_month, end_month, threshold, out_dir):
    df, pixel_lat, pixel_lon = point_series(lat, lon, year, start_month, end_month)
    print(f"Requested ({lat}, {lon}) -> nearest grid cell ({pixel_lat:.4f}, {pixel_lon:.4f})")

    tag = f"gust_compare_{year}_{pixel_lat:.4f}_{pixel_lon:.4f}"

    csv_path = out_dir / f"{tag}.csv"
    df.rename_axis("date").reset_index().assign(date=lambda d: d["date"].dt.date).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 5))
    for var, label, kw in SERIES:
        ax.plot(df.index, df[label], marker="o", markersize=2.5, linewidth=1.2, label=label, **kw)
    ax.axhline(threshold, color="red", linestyle=":", linewidth=1.5, label=f"{threshold:g} m/s lodging threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wind speed / gust (m s-1)")
    ax.set_title(f"Daily wind/gust estimates at ({pixel_lat:.4f}, {pixel_lon:.4f}), {year}")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_path = out_dir / f"{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Season peak of each estimate, plus whether it crossed the threshold.
    print(f"\nSeason max ({year}, months {start_month:02d}-{end_month:02d}):")
    for _, label, _ in SERIES:
        peak = df[label].max()
        print(f"  {label:32s} {peak:6.2f} m/s   exceeds {threshold:g}? {'yes' if peak > threshold else 'no'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=22.0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.lat, args.lon, args.year, args.start_month, args.end_month, args.threshold, Path(args.out_dir))
