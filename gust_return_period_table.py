#!/usr/bin/env python3
"""Wind gust return-period table, in the style of Pinera-Chavez et al. (2016)
Table 2 (p. 330) - "Seasonal wind gust speed return period for Obregon" -
but built from the full BARRA-C2 history at each paddock point instead of
their measured station record. Works on whichever gust variable the input
CSV holds (see extract_gust_history.py --variable).

Method: take the seasonal (Aug-Nov by default - `--start-month`/`--end-month`,
the growing-season stem-lodging risk window used elsewhere in this project,
see README) maximum gust for each year at a point, fit a Gumbel
(Fisher-Tippett Type I) distribution to those maxima - the standard
extreme-value model for annual wind extremes (the same family AS/NZS 1170.2
wind-loading return periods are built on) - then read off two things from the
fit:

  - Return level: the gust speed with annual exceedance probability 1/T,
    i.e. the speed expected to be equalled or exceeded on average once every
    T years. This is directly comparable to Pinera-Chavez Table 2.
  - Exceedance probability: for a given fixed gust speed, the probability
    that it is equalled or exceeded at least once within a T-year window,
    P = 1 - (1 - p)^T where p is the fitted annual exceedance probability.
    This answers "what's the chance we see an 20 m/s gust in the next 15
    years?" directly, which is the more intuitive way to read a range of
    threshold speeds (Pinera-Chavez's table only has one speed per period,
    split by lodging type; here it's one point per threshold x period cell).

Reads only the CSV written by extract_gust_history.py - no BARRA2 file
access, so this is fast and safe to re-run/tweak.

Example:
    python gust_return_period_table.py \\
        --csv /scratch/xe2/cb8590/wind-gusts2/gust_history_wsgsmax_197901-202601.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import gumbel_r

from barra_common import VARIABLES

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

RETURN_PERIODS_YEARS = [5, 10, 15, 20, 25]
GUST_THRESHOLDS_M_S = [22, 23, 24, 25, 26, 27]

POINT_LABELS = {
    "paddock_2024": "Site A (-35.0504, 147.3188)",
    "paddock_2025": "Site B (-35.0259, 147.3547)",
}


def tag_and_label(csv_path, variable):
    tag = csv_path.stem
    if tag.startswith("gust_history_"):
        tag = tag[len("gust_history_"):]
    if variable is None:
        variable = next((v for v in sorted(VARIABLES, key=len, reverse=True) if tag.startswith(v + "_")), None)
    label = VARIABLES[variable]["long_name"] if variable in VARIABLES else tag
    return tag, label


def seasonal_annual_max(df, start_month, end_month):
    """Max per point/year within [start_month, end_month] only (default
    Aug-Nov), not the full calendar year - matches plot_gust_climatology.py's
    seasonal_max. A year with no data in that window (e.g. the record's
    first/last partial year) simply drops out via the groupby."""
    df = df[df["date"].dt.month.between(start_month, end_month)].copy()
    df["year"] = df["date"].dt.year
    return df.groupby(["point", "year"], as_index=False)["gust_m_s"].max()


def fit_gumbel(annual_max_values):
    loc, scale = gumbel_r.fit(annual_max_values)
    return loc, scale


def return_levels(loc, scale, return_periods):
    non_exceedance_p = [1 - 1 / t for t in return_periods]
    levels = gumbel_r.ppf(non_exceedance_p, loc=loc, scale=scale)
    return dict(zip(return_periods, levels))


def exceedance_probabilities(loc, scale, thresholds, return_periods):
    """probs[threshold][T] = P(gust >= threshold at least once in T years)."""
    annual_p_exceed = {x: float(gumbel_r.sf(x, loc=loc, scale=scale)) for x in thresholds}
    probs = {}
    for x, p in annual_p_exceed.items():
        probs[x] = {t: 1 - (1 - p) ** t for t in return_periods}
    return probs, annual_p_exceed


def main(csv_path, out_dir, variable, start_month, end_month, return_periods, thresholds):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    tag, label = tag_and_label(csv_path, variable)
    season_tag = f"{tag}_m{start_month:02d}-{end_month:02d}"
    print(f"Variable: {label}")
    annual_max = seasonal_annual_max(df, start_month, end_month)

    level_rows = []
    prob_rows = []
    for point, g in annual_max.groupby("point"):
        n_years = len(g)
        loc, scale = fit_gumbel(g["gust_m_s"].values)
        point_label = POINT_LABELS.get(point, point)
        print(f"\n{point_label}: Gumbel fit to {n_years} years' month {start_month:02d}-{end_month:02d} maxima "
              f"({int(g['year'].min())}-{int(g['year'].max())}), loc={loc:.2f}, scale={scale:.2f}")

        levels = return_levels(loc, scale, return_periods)
        print(f"{'Return period (years)':>22} | {'Return-level gust (m/s)':>24}")
        for t in return_periods:
            print(f"{t:>22d} | {levels[t]:>24.1f}")
            level_rows.append(dict(point=point, point_label=point_label, return_period_years=t, return_level_gust_m_s=round(levels[t], 1)))

        probs, annual_p = exceedance_probabilities(loc, scale, thresholds, return_periods)
        header = f"{'Return period (years)':>22} | " + " | ".join(f"{x:>4g} m/s" for x in thresholds)
        print(header)
        for t in return_periods:
            row = " | ".join(f"{probs[x][t] * 100:>6.1f}%" for x in thresholds)
            print(f"{t:>22d} | {row}")
            prob_rows.append(dict(
                point=point, point_label=point_label, return_period_years=t,
                **{f"p_exceed_{x}ms_pct": round(probs[x][t] * 100, 1) for x in thresholds},
            ))

    levels_csv = out_dir / f"gust_return_levels_{season_tag}.csv"
    pd.DataFrame(level_rows).to_csv(levels_csv, index=False)
    print(f"\nWrote {levels_csv}")

    probs_csv = out_dir / f"gust_exceedance_probabilities_{season_tag}.csv"
    pd.DataFrame(prob_rows).to_csv(probs_csv, index=False)
    print(f"Wrote {probs_csv}")

    annual_max_csv = out_dir / f"gust_annual_max_{season_tag}.csv"
    annual_max.to_csv(annual_max_csv, index=False)
    print(f"Wrote {annual_max_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--variable", default=None, choices=list(VARIABLES), help="Overrides auto-detection from the CSV filename")
    parser.add_argument("--start-month", type=int, default=8, help="Season start month (default Aug)")
    parser.add_argument("--end-month", type=int, default=11, help="Season end month (default Nov)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--return-periods", type=int, nargs="+", default=RETURN_PERIODS_YEARS)
    parser.add_argument("--thresholds", type=float, nargs="+", default=GUST_THRESHOLDS_M_S)
    args = parser.parse_args()
    main(Path(args.csv), Path(args.out_dir), args.variable, args.start_month, args.end_month, args.return_periods, args.thresholds)
