#!/usr/bin/env python3
"""Two single-axis gust summaries with every GRDC cropping zone on one plot, in
the style of Papers/"min & max annual gust probabilities.png" but with each
zone drawn as its own named, state-coloured line instead of anonymous grey:

(a) Annual exceedance-probability curves - annual probability that the Aug-Dec
    (153-day) seasonal-maximum gust equals or exceeds a speed - for every zone.
(b) Average monthly-maximum gust climatology over the full calendar year (each
    month's maximum gust, averaged across all years), Aug-Dec shaded.

Each zone line is coloured by the state it mostly sits in, with a different
shade per zone within a state: QLD purples, NSW greens, SA oranges, WA blues.
State membership is by the state named first in the GRDC label, so the Qld/NSW
border zones ("NSW NE/Qld SE", "NSW NW/Qld SW") count as NSW. The two single-zone
states get fixed stand-alone colours so they don't clash: Tas Grain bright red,
Vic High Rainfall magenta. The two-Wagga-points average is a bold gold line.

The monthly plot smooths each line with a wrap-around moving average
(`--smooth-window`, default 3 months) and, unless `--no-ci`, shades a 95%
confidence interval of each month's mean; the annual plot's gust-speed axis is
set with `--xmin`/`--xmax`. `--stat` picks the per-zone daily statistic (mean).

Two `--mode`s:
  * zone (default): every zone is its `--stat` daily series.
  * envelope: also adds the whole-domain bounds as bold black lines - "Minimum -
    Australia" (the calmest cropping pixel anywhere each day) and "Maximum -
    Australia" (the windiest pixel anywhere). Needs the `_min`/`_max` columns
    (extract_zone_gust_history.py's default --stats).

Passing `--rain-zone-csv`/`--rain-point-csv` restricts every line to that
location's root-lodging-risk days (rain >= `--rain-threshold` mm on the day or
the previous `--rain-window-days`, judged from the zone `--rain-stat` rainfall).

WA Ord and WA Mallee are excluded (no wheat grown there). Reads only the CSVs -
no BARRA2 access.

Example (the "stem lodging risk" pair - all days):
    python plot_combined_gust_probabilities.py --stat mean --xmin 10 --xmax 30 --no-ci \\
        --zone-csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv \\
        --point-csv /scratch/xe2/cb8590/wind-gusts2/gust_history_wsgsmax_bias_197901-202601.csv
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("/scratch/xe2/cb8590/wind-gusts2")

EXCLUDE_ZONES = {"WA Ord", "WA Mallee"}
SEASON = (8, 12)  # Aug-Dec, the 153-day window
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# One sequential colormap per multi-zone state; a different shade per zone. The
# two Qld/NSW border zones ("NSW NE/Qld SE", "NSW NW/Qld SW") are counted as NSW
# (they start with "NSW"). Single-zone states get a fixed, clearly distinct
# colour instead of a one-shade ramp.
STATE_ORDER = ["QLD", "NSW", "SA", "WA"]
STATE_CMAP = {"QLD": "Purples", "NSW": "Greens", "SA": "Oranges", "WA": "Blues"}
FIXED_ZONE_COLORS = {"Tas Grain": "#e41a1c", "Vic High Rainfall": "#e7298a"}  # bright red / magenta
SHADE_LO, SHADE_HI = 0.40, 0.85  # keep hues recognisable, avoid the near-black ends

COL_WAGGA, COL_WAGGA_EDGE = "#d4b106", "#7a6800"  # gold - Wagga (2-point mean)
COL_ENVELOPE = "black"                            # domain min/max bounds
SEASON_SHADE = "#f0e6a8"                          # light yellow season band

def zone_state(name):
    """Multi-zone state a zone is counted under (the state named first in the
    GRDC label - so the Qld/NSW border zones "NSW .../Qld ..." count as NSW).
    Returns None for zones that get a fixed stand-alone colour."""
    if name in FIXED_ZONE_COLORS:
        return None
    for prefix, state in (("NSW", "NSW"), ("Qld", "QLD"), ("SA", "SA"), ("WA", "WA")):
        if name.startswith(prefix):
            return state
    return "SA"  # fallback (none expected)


def assign_zone_colors(zone_names):
    """Map each zone to a shade of its state's colormap (or its fixed colour);
    return also a state-grouped ordering for a tidy legend."""
    groups = {}
    for n in zone_names:
        st = zone_state(n)
        if st is not None:
            groups.setdefault(st, []).append(n)
    colors, order = {}, []
    for state in STATE_ORDER:
        members = sorted(groups.get(state, []))
        cmap = matplotlib.colormaps[STATE_CMAP[state]]
        for i, n in enumerate(members):
            frac = 0.5 * (SHADE_LO + SHADE_HI) if len(members) == 1 else \
                SHADE_LO + (SHADE_HI - SHADE_LO) * i / (len(members) - 1)
            colors[n] = cmap(frac)
            order.append(n)
    for n in zone_names:  # fixed-colour single zones, appended in input order
        if n in FIXED_ZONE_COLORS:
            colors[n] = FIXED_ZONE_COLORS[n]
            order.append(n)
    return colors, order


def zone_tag(csv_path):
    tag = csv_path.stem
    for sep in ("_summary_", "_medians_"):
        if sep in tag:
            return tag.split(sep, 1)[1]
    return tag


def seasonal_annual_max(series, start_month, end_month):
    """Aug-Dec maximum per year; only years with all season months present."""
    s = series[series.index.month.to_series(index=series.index).between(start_month, end_month)]
    by_year = s.groupby(s.index.year)
    n_months = by_year.apply(lambda g: g.index.month.nunique())
    complete = n_months[n_months == (end_month - start_month + 1)].index
    return by_year.max().loc[complete]


def monthly_max_stats(series):
    """Per calendar month (1..12): the mean across years of that month's maximum
    gust, and the standard error of that mean. Returns (mean, sem) as arrays of
    length 12 aligned to months 1..12 (NaN where a month has no data)."""
    monthly = series.groupby([series.index.year, series.index.month]).max()
    monthly.index.names = ["year", "month"]
    g = monthly.groupby("month")
    idx = np.arange(1, 13)
    return g.mean().reindex(idx).values, g.sem().reindex(idx).values


def circular_smooth(values, window):
    """Moving average over the 12 months with Dec<->Jan wrap-around. window=1
    (or 0) returns the values unchanged. NaNs are ignored within each window."""
    if window is None or window <= 1:
        return np.asarray(values, dtype="float64")
    k = window // 2
    ext = np.concatenate([values[-k:], values, values[:k]])
    out = np.full(12, np.nan)
    for i in range(12):
        win = ext[i:i + window]
        if np.isfinite(win).any():
            out[i] = np.nanmean(win)
    return out


def exceedance_curve(annual_max, thresholds):
    return (annual_max.values[:, None] >= thresholds[None, :]).mean(axis=0)


def load_zone_stat(zone_csv, stat):
    """{zone: daily Series} for the given per-zone statistic, excluding EXCLUDE_ZONES."""
    zdf = pd.read_csv(zone_csv, parse_dates=["date"], index_col="date")
    suffix = f"_{stat}"
    out = {}
    for col in zdf.columns:
        if col.endswith(suffix):
            name = col[: -len(suffix)]
            if name not in EXCLUDE_ZONES:
                out[name] = zdf[col]
    if not out:
        raise SystemExit(f"No columns ending in {suffix!r} in {zone_csv}")
    return out


def load_wagga(point_csv):
    pdf = pd.read_csv(point_csv, parse_dates=["date"])
    value_col = "gust_m_s" if "gust_m_s" in pdf.columns else pdf.columns[-1]
    wide = pdf.pivot_table(index="date", columns="point", values=value_col)
    wagga = wide.mean(axis=1)  # average of the two paddock points
    wagga.index = pd.to_datetime(wagga.index)
    return wagga


# --- Root-lodging rain filtering ------------------------------------------
# A day is "at risk of root lodging" if the soil is likely still wet: it rained
# at least `threshold` mm that day, or on any of the previous `window_days`
# days (the rain day plus the days after it). Restricting the gust series to
# these days answers "how strong are the gusts when the ground is wet?".

def at_risk_mask(rain, threshold, window_days):
    wet = rain >= threshold
    mask = wet.copy()
    for k in range(1, window_days + 1):
        mask = mask | wet.shift(k, fill_value=False)  # rows are consecutive days
    return mask


def load_zone_rain(rain_zone_csv, stat="mean"):
    """{zone: daily rain Series}. Accepts either a per-zone summary CSV (columns
    "<zone>_<stat>", from which `stat` selects the zone's daily rainfall) or a
    plain medians-style CSV whose columns are already zone names."""
    df = pd.read_csv(rain_zone_csv, parse_dates=["date"], index_col="date")
    suffix = f"_{stat}"
    cols = [c for c in df.columns if c.endswith(suffix)]
    if cols:
        return {c[: -len(suffix)]: df[c] for c in cols if c[: -len(suffix)] not in EXCLUDE_ZONES}
    return {z: df[z] for z in df.columns if z not in EXCLUDE_ZONES}


def load_wagga_rain(rain_point_csv):
    pdf = pd.read_csv(rain_point_csv, parse_dates=["date"])
    value_col = "rainfall_mm" if "rainfall_mm" in pdf.columns else pdf.columns[-1]
    wide = pdf.pivot_table(index="date", columns="point", values=value_col)
    rain = wide.mean(axis=1)
    rain.index = pd.to_datetime(rain.index)
    return rain


def legend_outside(ax):
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def plot_annual_probability(zones, colors, order, highlights, title, out_path, xlim=None):
    sm, em = SEASON
    zone_ann = {z: seasonal_annual_max(s, sm, em) for z, s in zones.items()}
    hi_ann = [(seasonal_annual_max(s, sm, em), style) for s, style in highlights]
    overall_max = max([a.max() for a in zone_ann.values()] + [a.max() for a, _ in hi_ann])
    thresholds = np.arange(0, int(math.ceil(overall_max / 5.0) * 5) + 1, 1.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for z in order:
        ax.plot(thresholds, exceedance_curve(zone_ann[z], thresholds), "-",
                color=colors[z], linewidth=1.4, zorder=2, label=z)
    for a, style in hi_ann:
        ax.plot(thresholds, exceedance_curve(a, thresholds), **style)

    ax.set_xlim(*(xlim if xlim is not None else (thresholds[0], thresholds[-1])))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Wind gust speed (m/s)")
    ax.set_ylabel("Annual probability")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(title)
    legend_outside(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_monthly_climatology(zones, colors, order, highlights, title, out_path,
                             smooth_window=3, ci=True):
    months = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axvspan(SEASON[0] - 0.5, SEASON[1] + 0.5, color=SEASON_SHADE, alpha=0.5, zorder=0,
               label="Aug–Dec")

    # Unified list of (series, style) - zone lines first, then Wagga/highlights.
    lines = [(zones[z], dict(color=colors[z], linewidth=1.4, zorder=2, label=z)) for z in order]
    lines += [(s, style) for s, style in highlights]

    for series, style in lines:
        mean, sem = monthly_max_stats(series)
        line = circular_smooth(mean, smooth_window)
        color = style["color"]
        if ci:
            half = 1.96 * sem  # 95% CI of the mean
            lo = circular_smooth(mean - half, smooth_window)
            hi = circular_smooth(mean + half, smooth_window)
            ax.fill_between(months, lo, hi, color=color, alpha=0.12, linewidth=0,
                            zorder=style.get("zorder", 2) - 0.5)
        ax.plot(months, line, color=color, linewidth=style.get("linewidth", 1.4),
                linestyle=style.get("linestyle", "-"), zorder=style.get("zorder", 2),
                marker=style.get("marker"), markersize=style.get("markersize", 5),
                markerfacecolor=style.get("markerfacecolor"), markeredgecolor=style.get("markeredgecolor"),
                label=style.get("label"))

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind gust speed (m/s)")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(title)
    legend_outside(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(zone_csv, point_csv, out_dir, mode, stat,
         rain_zone_csv=None, rain_point_csv=None, rain_threshold=10.0, rain_window_days=3, rain_stat="mean",
         xmin=None, xmax=None, smooth_window=3, ci=True):
    wagga = load_wagga(point_csv)
    tag = zone_tag(zone_csv)
    sm, em = SEASON

    zones = load_zone_stat(zone_csv, stat)
    colors, order = assign_zone_colors(list(zones))
    zmin = load_zone_stat(zone_csv, "min") if mode == "envelope" else {}
    zmax = load_zone_stat(zone_csv, "max") if mode == "envelope" else {}

    rain_filter = rain_zone_csv is not None
    rain_tag = ""
    if rain_filter:
        # Restrict each location's gusts to its own root-lodging-risk days,
        # using that location's rainfall (zone median / Wagga 2-point mean).
        zone_rain = load_zone_rain(rain_zone_csv, rain_stat)
        zmasks = {z: at_risk_mask(zone_rain[z].reindex(zones[z].index), rain_threshold, rain_window_days)
                  for z in zones}
        zones = {z: s.where(zmasks[z]) for z, s in zones.items()}
        wagga_rain = load_wagga_rain(rain_point_csv)
        wagga = wagga.where(at_risk_mask(wagga_rain.reindex(wagga.index), rain_threshold, rain_window_days))
        zmin = {z: s.where(zmasks[z]) for z, s in zmin.items()}
        zmax = {z: s.where(zmasks[z]) for z, s in zmax.items()}
        rain_tag = f"_rain{rain_threshold:g}mm{rain_window_days}d"

    wagga_style = dict(color=COL_WAGGA, markerfacecolor=COL_WAGGA, markeredgecolor=COL_WAGGA_EDGE,
                       marker="o", markersize=5, linewidth=2.2, zorder=6,
                       label="Wagga Wagga, NSW")
    highlights = [(wagga, wagga_style)]

    if mode == "envelope":
        dmin = pd.concat(zmin, axis=1).min(axis=1)  # calmest pixel among (at-risk) zones
        dmax = pd.concat(zmax, axis=1).max(axis=1)  # windiest pixel among (at-risk) zones
        highlights += [
            (dmax, dict(color=COL_ENVELOPE, marker="o", markersize=4, linewidth=2.4, linestyle="-",
                        zorder=5, label="Maximum – Australia (windiest pixel)")),
            (dmin, dict(color=COL_ENVELOPE, marker="o", markersize=4, linewidth=2.4, linestyle="--",
                        zorder=5, label="Minimum – Australia (calmest pixel)")),
        ]
        name = f"envelope_{stat}"
    else:
        name = stat
    name += rain_tag

    # Non-rain plots are the gust-only "stem lodging" risk; the rain-filtered
    # (wet-soil) plots are the "root lodging" risk.
    lodging = "root" if rain_filter else "stem"
    xlim = None if xmin is None and xmax is None else (xmin, xmax)
    plot_annual_probability(
        zones, colors, order, highlights,
        f"Annual {lodging} lodging risk",
        out_dir / f"gust_combined_annual_prob_{name}_{tag}_m{sm:02d}-{em:02d}.png",
        xlim=xlim,
    )
    plot_monthly_climatology(
        zones, colors, order, highlights,
        f"Monthly {lodging} lodging risk",
        out_dir / f"gust_combined_monthly_max_{name}_{tag}.png",
        smooth_window=smooth_window, ci=ci,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zone-csv", required=True)
    parser.add_argument("--point-csv", required=True)
    parser.add_argument("--mode", default="zone", choices=["zone", "envelope"])
    parser.add_argument("--stat", default="mean", choices=["min", "mean", "median", "p75", "p90", "max"],
                        help="per-zone daily statistic for the zone lines (default mean)")
    parser.add_argument("--rain-zone-csv", default=None,
                        help="rain_zone_medians CSV; enables the root-lodging-risk-day filter")
    parser.add_argument("--rain-point-csv", default=None, help="rain_history (pr) CSV for the Wagga points")
    parser.add_argument("--rain-threshold", type=float, default=10.0, help="mm/day that counts as a wet day")
    parser.add_argument("--rain-window-days", type=int, default=3, help="risk days after each wet day")
    parser.add_argument("--rain-stat", default="mean", choices=["min", "mean", "median", "p75", "p90", "max"],
                        help="per-zone rainfall statistic used to judge wetness (default mean)")
    parser.add_argument("--xmin", type=float, default=None, help="annual plot: lower gust-speed axis bound (m/s)")
    parser.add_argument("--xmax", type=float, default=None, help="annual plot: upper gust-speed axis bound (m/s)")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="monthly plot: months in the wrap-around moving average (1 = off)")
    parser.add_argument("--no-ci", dest="ci", action="store_false", help="monthly plot: drop the 95%% CI band")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.zone_csv), Path(args.point_csv), Path(args.out_dir), args.mode, args.stat,
         args.rain_zone_csv, args.rain_point_csv, args.rain_threshold, args.rain_window_days, args.rain_stat,
         args.xmin, args.xmax, args.smooth_window, args.ci)
