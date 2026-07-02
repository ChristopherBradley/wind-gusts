#!/usr/bin/env python3
"""Overlay the different daily wind/gust estimates at a single lat/lon point for
one season, so the model gust (wsgsmax) and the Pinera-Chavez empirical gust
(reconstructed from sfcWindmax) can be compared side by side. Plots four daily
series over the season (Aug-Nov by default), ordered here (and in the legend)
to match their typical position on the plot, largest first:

    Pinera-Chavez gust - the mean gustified at 10 m via Berry/Pinera Eq. 6
                         (Pinera-Chavez et al. 2016, p. 328; gust-factor ~2.97),
                         then brought to 2 m crop height (x0.575)
    Corrected gust     - wsgsmax: the model gust, bias-adjusted only (x0.90),
                         still at 10 m - deliberately NOT height-corrected, so
                         it is directly comparable to the Pinera-Chavez gust
                         at the same (10 m) stage of that derivation
    Raw maximum        - sfcWindmax: daily-max hourly-MEAN wind at 10 m (as-is)
    Corrected maximum  - the raw maximum brought to 2 m crop height (x0.575)

Pinera-Chavez gust and Corrected gust land close together - the interesting
comparison, since one is the model's own (bias-corrected) gust and the other
is entirely independent, reconstructed from the mean via an empirical gust
factor (see barra_common.py for why). Raw maximum and Corrected maximum are
the underlying hourly-mean winds, shown for context; Corrected maximum is the
smallest of the four since it is a mean (no gust factor) at the slower,
near-ground 2 m height.

Every factor lives in barra_common.VARIABLES (single source of truth); this
script only reads the two on-disk variables (sfcWindmax, wsgsmax) once each and
applies each logical variable's `convert` to the extracted point series. Writes
one PNG and one CSV (one column per estimate). The PNG marks both 25-year
lodging thresholds from Pinera-Chavez et al. 2016, Table 2, p. 330 - 22 m/s
(stem) and 18 m/s (root) - as dotted horizontal lines, and shades the column
behind any day whose highest series that day crosses one of them (red for
stem, orange for root-only).

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

# (logical variable, legend label, matplotlib line kwargs), ordered to match
# the plot top-to-bottom by season peak: Pinera-Chavez gust ~= Corrected gust
# > Raw maximum > Corrected maximum. Mean-wind quantities are blue, the
# Pinera-Chavez reconstruction green, the model's bias-corrected gust orange;
# every line is solid except Corrected maximum (dashed), so the one
# intermediate, non-headline quantity stands out from the rest.
SERIES = [
    ("wsgsmax_bias", "max-gust * 0.9 (10 m)", dict(color="tab:orange", linestyle="-")),
    ("pinera_gust", "max-wind gust corrected (2 m)", dict(color="tab:green", linestyle="-")),
    ("sfcWindmax", "max-wind raw (10 m)", dict(color="tab:blue", linestyle="-")),
    ("sfcWindmax_2m", "max-wind height corrected (2 m)", dict(color="tab:blue", linestyle="--")),
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
    # df.index.name = "date"
    return df, pixel_lat, pixel_lon


def _shade_exceedance_days(ax, df, threshold, root_threshold):
    """Shade a one-day-wide column behind each date whose highest series that
    day crosses a lodging threshold: red where it crosses the stem threshold,
    orange where it crosses only the (lower) root threshold. Only one series
    needs to cross for the day to count, since the two gust estimates are the
    ones actually meant to be compared to these thresholds."""
    half_day = pd.Timedelta(hours=12)
    day_max = df.max(axis=1)
    stem_label, root_label = "exceeds 22 m/s", "exceeds 18 m/s"
    for date, peak in day_max.items():
        if peak > threshold:
            ax.axvspan(date - half_day, date + half_day, color="red", alpha=0.12, linewidth=0, zorder=0, label=stem_label)
            stem_label = None
        elif peak > root_threshold:
            ax.axvspan(date - half_day, date + half_day, color="orange", alpha=0.15, linewidth=0, zorder=0, label=root_label)
            root_label = None


def main(lat, lon, year, start_month, end_month, threshold, root_threshold, out_dir):
    df, pixel_lat, pixel_lon = point_series(lat, lon, year, start_month, end_month)
    print(f"Requested ({lat}, {lon}) -> nearest grid cell ({pixel_lat:.4f}, {pixel_lon:.4f})")

    tag = f"gust_compare_{year}_{pixel_lat:.4f}_{pixel_lon:.4f}"

    csv_path = out_dir / f"{tag}.csv"
    df.rename_axis("date").reset_index().assign(date=lambda d: d["date"].dt.date).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_exceedance_days(ax, df, threshold, root_threshold)
    for var, label, kw in SERIES:
        ax.plot(df.index, df[label], marker="o", markersize=2.5, linewidth=1.2, label=label, **kw)
    ax.axhline(threshold, color="red", linestyle=":", linewidth=1.5)
    ax.axhline(root_threshold, color="darkorange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wind speed / gust (m s-1)")
    ax.set_title(f"Daily wind/gust estimates at ({pixel_lat:.4f}, {pixel_lon:.4f}), {year}")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_path = out_dir / f"{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Season peak of each estimate, plus whether it crossed either threshold.
    print(f"\nSeason max ({year}, months {start_month:02d}-{end_month:02d}):")
    for _, label, _ in SERIES:
        peak = df[label].max()
        exceeds_stem = "yes" if peak > threshold else "no"
        exceeds_root = "yes" if peak > root_threshold else "no"
        print(
            f"  {label:32s} {peak:6.2f} m/s   exceeds {threshold:g}? {exceeds_stem:3s}"
            f"   exceeds {root_threshold:g}? {exceeds_root}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--end-month", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=22.0, help="Stem-lodging gust threshold, m/s")
    parser.add_argument("--root-threshold", type=float, default=18.0, help="Root-lodging gust threshold, m/s")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(
        args.lat,
        args.lon,
        args.year,
        args.start_month,
        args.end_month,
        args.threshold,
        args.root_threshold,
        Path(args.out_dir),
    )
