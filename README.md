# wind-gusts2

Analysis of damaging wind gusts over Australian winter-cereal cropping land,
using [BARRA2](https://www.bom.gov.au/research/projects/reanalysis/) AUST-04
regional reanalysis data. The core question: how often (and where) does the
daily-max wind gust exceed a damaging threshold (22 m/s) within
cropping zones during the Aug-Nov growing season, and how has that changed
2017-2025?

## The wind gust variable and adjustments

The 22 m/s threshold is the gust a wheat stem must withstand to hold a 1-in-25-year
stem-lodging return period (Pinera-Chavez et al. 2016, Table 2). It is a *gust* at
~2 m crop height, not a mean wind speed. We therefore read BARRA-C2's modelled gust
`wsgsmax` (standard_name `wind_speed_of_gust`: daily max of the 3-s peak gust at 10 m),
**not** `sfcWindmax`, which is only the daily max of the *hourly-mean* wind - the
quantity Pinera-Chavez had to gustify by hand because they lacked gust observations.
Using `wsgsmax` means we do not reproduce their empirical mean-to-gust conversion at
all; the model already resolves a physical gust.

Two corrections are applied to `wsgsmax` (in `barra_common.py`, baked into the
variable's `convert` so every script sees the adjusted value) to make it comparable
to the 22 m/s crop-height threshold:

- **Bias: x0.90.** Bush et al. (2025) / BoM (2025) report a known positive bias in
  the BARRA2 gust diagnostic from a mis-set constant; the correct value reduces the
  gust by ~10%.
- **Height 10 m -> 2 m: x0.575.** `wsgsmax` is a 10 m gust; the threshold is at the
  ~2 m pertinent height for crop lodging. We scale with the neutral-log wind profile
  over a wheat canopy, reusing the Berry et al. (2003b) / Pinera-Chavez roughness
  parameters (displacement d = 0.75 h, roughness z0 = (h - d)/3 for h = 1 m).

Combined multiplier ~0.518. Both factors are named constants at the top of the
`VARIABLES` block in `barra_common.py` - set either to 1.0 to disable. Note the height
profile strictly describes the mean wind, so applying it to the model gust is an
approximation; it is also a large correction, so exceedances of 22 m/s are far rarer
than with unadjusted wind. See `Papers/` for the two source PDFs.

### Comparing against the Pinera-Chavez empirical gust

As a cross-check we can also do what Pinera-Chavez actually did - reconstruct a gust
from the mean wind (`sfcWindmax`) instead of using the model gust. Their Eq. 6
(Pinera-Chavez et al. 2016, p. 328; after Berry et al. 2003b) turns an hourly-mean
wind into a peak 0.3 s gust with a constant gust factor `1 + 0.42*TI*ln(3600/tau)
~= 2.97` (turbulence intensity `TI = 0.5` for wind over a wheat crop). We apply that
gust factor first, at the mean's native 10 m (the same height as the model's own raw
gust, so the two are directly comparable there), then bring the result down to 2 m
with the same height factor used for the model gust: `pinera_gust = sfcWindmax *
2.97 * 0.575 ~= sfcWindmax * 1.71`.

This is exposed as extra *derived* variables (see below) purely for comparison - the
main pipeline still uses the model gust `wsgsmax`. The two 2 m gust estimates diverge a
lot (the Pinera one assumes very high near-canopy turbulence, so its 2 m gust is ~1.8x
the height-corrected model gust); that divergence is the thing to look at. The constants
`PINERA_TURBULENCE_INTENSITY`, `PINERA_GUST_DURATION_S` etc. are all in `barra_common.py`.

Use `compare_point_gusts.py` (step 5 below) to plot all of these together at a point.

The pipeline masks BARRA2 to pixels that are (a) classified as winter cereal
cropping land in the National Land Use Map (NLUM) and (b) inside a GRDC
cropping region, then summarises wind speed/direction for those pixels as
GeoTIFFs, CSVs, and plots.

## Environment

```
conda activate shelterbelts
```

Key dependencies: `xarray`, `rioxarray`, `rasterio`, `geopandas`, `pyproj`,
`numpy`, `pandas`, `matplotlib`.

Runs on an NCI compute node with no outbound internet access and a hard
1800 CPU-second budget per process on a single CPU - this is why several
scripts process one year at a time rather than looping over years.

To run on NCI (gadi): from a session with `/g/data/ob53` and `/scratch/xe2`
in the storage flags,

```
cd /home/147/cb8590/Projects/wind-gusts2
git pull
conda activate shelterbelts
python <script.py> ...   # see the Workflow section below
```

Because the compute nodes have no internet, `git pull` must be done from a
login node (or any node with outbound access) before starting the compute
job.

Paths (hardcoded as defaults throughout the scripts):
- Code: `/home/147/cb8590/Projects/wind-gusts2`
- Outputs: `/scratch/xe2/cb8590/wind-gusts2`
- BARRA2 source data: `/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day`
  (see `/g/data/ob53/BARRA2/README.txt` for the dataset itself)

`barra_common.py` is shared plumbing (not run directly): it knows how to find/open
BARRA2 monthly files for a given variable, convert units, attach the correct
CRS for GeoTIFF export, build land-use/zone masks, and write colourised uint8
GeoTIFFs. The `VARIABLES` dict there is the registry of selectable `--variable`
names, and is the single source of truth for every adjustment factor. Each entry
has a `convert` function (raw model field -> analysis quantity) and, for *derived*
variables, a `source` giving the on-disk BARRA variable to actually read.
`open_variable` reads `source` and applies `convert`, so a derived variable needs
no file of its own. Supported names:

| `--variable`     | reads (`source`) | what it is |
|------------------|------------------|------------|
| `wsgsmax`        | wsgsmax          | **model gust, bias- & height-adjusted to 2 m** (the primary analysis quantity) |
| `wsgsmax_raw`    | wsgsmax          | model gust, raw at 10 m (no adjustment) |
| `wsgsmax_bias`   | wsgsmax          | model gust, bias-adjusted only (x0.90), still at 10 m (no height correction) |
| `sfcWindmax`     | sfcWindmax       | daily-max hourly-*mean* wind at 10 m (raw) |
| `sfcWindmax_2m`  | sfcWindmax       | that mean, height-corrected to 2 m |
| `pinera_gust`    | sfcWindmax       | 2 m gust reconstructed from the mean via Pinera-Chavez Eq. 6 |
| `tasmax`         | tasmax           | daily max temperature (K -> degC) |
| `uas` / `vas`    | uas / vas        | daily mean eastward / northward wind (for direction) |

The `wsgsmax_raw`, `wsgsmax_bias`, `sfcWindmax_2m` and `pinera_gust` entries
exist only for the point comparison (`compare_point_gusts.py`); the pipeline
proper uses `wsgsmax`.

## Workflow

### 1. Build the cropping land-use mask (once)

Resamples the high-resolution NLUM land-use raster onto the BARRA2 grid,
thresholds it, and intersects it with the GRDC cropping-zone polygons.
Caches the result so later per-year scripts don't repeat the expensive
reprojection.

```
python build_landuse_mask.py --threshold 5000 \
    --landuse-tif NLUM_v7_probSurf_2021_331_5_W_CER.tif \
    --zones-gpkg grdc_cropping_zones.gpkg
```

Writes `landuse_mask.tif` to the output dir.

### 2. Find wind exceedances, one year at a time

For each pixel/date in the mask where the adjusted `wsgsmax` gust exceeds a
threshold, records wind direction (from daily-mean `uas`/`vas`) and GRDC
cropping region. Defaults to the Aug-Nov window (`--start-month 8
--end-month 11`).

```
python wind_exceedances_year.py --year 2022 --threshold 22
```

Writes `wind_exceedances_<year>.csv` (gust column `max_gust`). Run once per
year, then combine:

```
python combine_exceedances.py --start-year 2017 --end-year 2025
```

Writes the combined CSV/GeoPackage plus a day-aggregated CSV (one row per
exceedance day, picking a representative pixel/coordinate per day).

### 3. Per-year masked wind summaries

Takes the seasonal max of a variable, restricted to the cropping mask;
writes a GeoTIFF and caches the masked pixel values for box-plotting.

```
python landuse_wind_year.py --variable wsgsmax --year 2022 --start-month 8 --end-month 11
```

Then build a box plot across years from the cached `.npy` files:

```
python landuse_wind_boxplot.py --variable wsgsmax --start-year 2017 --end-year 2025 --line 22
```

### 4. Australia-wide (unmasked) summaries

Seasonal/monthly max GeoTIFFs over the whole BARRA2 domain, no land-use
masking:

```
python make_monthly_max_geotiff.py --variable tasmax --year 2022 --month 9
python make_seasonal_max_geotiff.py --variable wsgsmax --start-year 2017 --end-year 2025 --start-month 8 --end-month 11
```

### 5. Point/pixel time series

Daily time series (CSV + plot) of a variable at a given lat/lon:

```
python extract_point_timeseries.py --variable tasmax --lat -36.456671 --lon 148.262724 \
    --start-year 2022 --start-month 8 --end-year 2022 --end-month 11
```

**Specific paddock point-years (run these two).** Two paddocks near Wagga Wagga,
NSW - one for 2024, one for 2025. `compare_point_gusts.py` overlays four
wind/gust estimates (Pinera-Chavez gust, corrected gust, raw maximum, corrected
maximum) for the Aug-Nov season on one plot + CSV, with the 22 m/s stem- and
18 m/s root-lodging lines, so you can see how the model gust and the Pinera
reconstruction compare:

```
# 2024 paddock (-35.050418, 147.318795)
python compare_point_gusts.py --lat -35.050418 --lon 147.318795 --year 2024

# 2025 paddock (-35.025873, 147.354694)
python compare_point_gusts.py --lat -35.025873 --lon 147.354694 --year 2025
```

Writes `gust_compare_<year>_<lat>_<lon>.png` / `.csv`. To plot a single estimate
on its own instead, `extract_point_timeseries.py` takes any one `--variable` (e.g.
`--variable wsgsmax` or `--variable pinera_gust`).

Full-year time series at each year's worst exceedance pixel (from the
day-aggregated CSV produced in step 2):

```
python top_pixel_timeseries.py --by-day-csv wind_exceedances_2017-2025_by_day.csv
```

### 6. Full-history climatology and return-period table at the two paddock points

Daily gust at the two paddock points, over every month of BARRA-C2 available
on disk (currently 1979-01 to 2026-01), for either `wsgsmax_bias` ("Corrected
gust" = `wsgsmax * 0.9`, still at 10 m) or `wsgsmax` (bias- AND
height-adjusted to 2 m, the main pipeline's gust). This reads ~565 monthly
files and takes a few minutes even processing one calendar year at a time,
so it runs as a PBS job (`extract_gust_history.pbs`, ~0.6 SU per variable):

```
qsub -v VARIABLE=wsgsmax_bias extract_gust_history.pbs   # 10 m, bias-only ("Corrected gust")
qsub -v VARIABLE=wsgsmax      extract_gust_history.pbs   # 2 m, bias- and height-adjusted (main pipeline gust)
```

Each writes one long-format CSV (`gust_history_<variable>_<start>-<end>.csv`,
one row per point/day) - the single source of truth for the two analyses
below, which only read this CSV and never touch the BARRA2 files again:

```
python plot_gust_climatology.py --csv /scratch/xe2/cb8590/wind-gusts2/gust_history_wsgsmax_bias_197901-202601.csv
python gust_return_period_table.py --csv /scratch/xe2/cb8590/wind-gusts2/gust_history_wsgsmax_bias_197901-202601.csv
```

`plot_gust_climatology.py` writes a monthly-max CSV/plot (every month, full
year) and a seasonal-max CSV/plot: one value per year, the max within
`--start-month`/`--end-month` (default Aug-Nov, the growing-season
stem-lodging window used elsewhere in this project - not the full calendar
year). `gust_return_period_table.py` fits a Gumbel distribution to each
point's Aug-Nov seasonal maxima (same `--start-month`/`--end-month` window,
not the full calendar year - a year's overall worst gust often comes from
outside the growing season, so this matters: at these points it moves the
Gumbel location down by ~1.5-1.7 m/s vs. a full-year fit) and writes, in the
style of Pinera-Chavez et al. (2016) Table 2: T-year return-level gust speeds
for T in {5,10,15,20,25} years, rounded to 1 dp, and the probability of a
gust of at least {22,23,24,25,26,27} m/s occurring within each of those
windows (values rounded to 1 dp; the default thresholds were moved up from
18-22 to 22-27 m/s for `wsgsmax_bias`, since 18-22 m/s all round to ~100%
there - see below). `plot_gust_return_period_heatmap.py` turns the
exceedance-probability CSV into a two-panel heatmap (one panel per site,
shared red<->blue diverging colour scale - red high/bad, blue low/good):

```
python plot_gust_return_period_heatmap.py --csv /scratch/xe2/cb8590/wind-gusts2/gust_exceedance_probabilities_wsgsmax_bias_197901-202601_m08-11.csv
```

The two gust variables tell very different stories at these points:
`wsgsmax_bias` (10 m) Aug-Nov return levels run ~23-27 m/s, mostly above the
22-27 m/s threshold range, so its exceedance-probability table/heatmap is
red-heavy (high probability). `wsgsmax` (2 m) return levels run only ~13-16
m/s, well below that range - its exceedance probabilities would all be near
0% at these thresholds (still using the original 18-22 m/s default if
re-run). This matches what `compare_point_gusts.py` already showed: the 10 m
bias-only gust and the 2 m Pinera-Chavez reconstruction (`pinera_gust`) land
close together, not the model's own 2 m gust - the height correction dominates
the difference between the two tables here.

### 7. Per-GRDC-zone gust history, exceedance probabilities and return levels

The step-6 analysis at the two paddock points, generalised to all 18 GRDC
cropping zones in `grdc_cropping_zones.gpkg`. `extract_zone_gust_history.py`
walks the full BARRA-C2 history one month at a time and, for every day and
every zone, summarises all pixels whose centre falls in that zone's polygon
with four statistics - median, upper quartile (p75), p90 and maximum. It reads
the whole Australia grid (not a handful of points), so it is much heavier than
step 6 and is meant for the PBS job:

```
qsub extract_zone_gust_history.pbs   # VARIABLE defaults to wsgsmax_bias
```

This writes two CSVs: `gust_zone_summary_<tag>.csv` (date + 72 columns, the
four stats grouped per zone as `<zone>_median`/`_p75`/`_p90`/`_max`) and the
convenience subset `gust_zone_medians_<tag>.csv` (date + one median column per
zone). The same PBS job then also regenerates the seasonal-max GeoTIFF over the
whole domain for `wsgsmax_bias` (= `wsgsmax * 0.9`), Aug-Nov, last five seasons
(2021-2025), `--direct-units` so pixel values are wind speed in m/s -
`wsgsmax_bias_seasonal_max_2021-2025_m08-11.tif` (rerun `make_seasonal_max_geotiff.py`
with an earlier `--start-year` for a 10/15/20/25-year window).

Both analyses below only read those CSVs - no BARRA2 access - so they are fast
and safe to re-run/tweak:

```
python plot_zone_gust_probabilities.py --stat median --xmin 10 --xmax 30 \
    --csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv
python plot_zone_gust_probabilities.py --stat max \
    --csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv
python zone_gust_25yr_return_level.py --stat median \
    --csv /scratch/xe2/cb8590/wind-gusts2/gust_zone_summary_wsgsmax_bias_197901-202601.csv
```

`plot_zone_gust_probabilities.py` builds, per zone, the empirical *annual
probability* that the Aug-Nov seasonal-maximum gust equals or exceeds a range
of speeds (the fraction of years in the record whose seasonal max is >= each
speed - no distribution fitted; runs from 1.0 at low speeds to 0.0 at high),
in the style of `Papers/Wind gust probabilities.png`. `--stat` picks which
daily zone statistic to build the curve from (`median`, the typical gust
across a zone, or `max`, the windiest pixel each day); `--xmin`/`--xmax`
truncate the speed axis. It writes a threshold x zone table CSV, one figure per
zone, and a combined 6x3 panel (every panel independently labelled).

`zone_gust_25yr_return_level.py` reads off, per zone, the gust speed with an
annual exceedance probability of 0.04 - the empirical 1-in-25-year gust, i.e.
where each curve crosses annual probability 0.04 (the 0.96 quantile of that
zone's annual maxima). It writes a table CSV and, unless `--no-gpkg`, adds the
values back onto `grdc_cropping_zones.gpkg` as a real-valued attribute
(default column `gust25yr_median_ms`, keyed by `AEZ`).

## Outputs

All scripts write to `/scratch/xe2/cb8590/wind-gusts2` by default
(override with `--out-dir`/`--out-path`). GeoTIFFs are uint8 with an
embedded GDAL colour table so they auto-colour when opened in QGIS.
