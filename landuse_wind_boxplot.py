#!/usr/bin/env python3
"""Combine the per-year .npy pixel-value files written by landuse_wind_year.py
into a single box plot (one box per year, outliers included), with a
horizontal reference line at a given threshold.

Example:
    python landuse_wind_boxplot.py --variable wsgsmax --start-year 2017 --end-year 2025 --line 22
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barra_common import VARIABLES

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")


def main(variable, start_year, end_year, line_value, out_dir):
    cfg = VARIABLES[variable]
    years = list(range(start_year, end_year + 1))

    yearly_values = {}
    for year in years:
        npy_path = out_dir / f"{variable}_landuse_values_{year}.npy"
        yearly_values[year] = np.load(npy_path)

    n_pixels = {len(v) for v in yearly_values.values()}
    n_pixels_label = n_pixels.pop() if len(n_pixels) == 1 else "varies"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([yearly_values[y] for y in years], tick_labels=[str(y) for y in years], showfliers=True)
    ax.axhline(line_value, color="red", linestyle=":", linewidth=1.5, label=f"{line_value:g} {cfg['units_out']}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Aug-Nov max {cfg['long_name'].lower()} ({cfg['units_out']})")
    ax.set_title(f"Max {cfg['long_name'].lower()} per pixel in winter cereal cropping pixels (n={n_pixels_label})")
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / f"{variable}_landuse_boxplot_{start_year}-{end_year}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    for year in years:
        v = yearly_values[year]
        print(f"{year}: n={v.size} median={np.median(v):.2f} max={v.max():.2f} n_exceeding_{line_value:g}={(v > line_value).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="wsgsmax", choices=list(VARIABLES))
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--line", type=float, default=22.0, help="Reference horizontal line value on the box plot")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(args.variable, args.start_year, args.end_year, args.line, Path(args.out_dir))
