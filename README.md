# wind-gusts2

Analysis of damaging wind gusts over Australian winter-cereal cropping land,
using [BARRA2](https://www.bom.gov.au/research/projects/reanalysis/) AUST-04
regional reanalysis data. The core question: how often (and where) does
daily-max wind speed exceed a damaging threshold (e.g. 22 m/s) within
cropping zones during the Sep-Nov growing season, and how has that changed
2017-2025?

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

Paths (hardcoded as defaults throughout the scripts):
- Code: `/home/147/cb8590/Projects/wind-gusts2`
- Outputs: `/scratch/xe2/cb8590/wind-gusts2`
- BARRA2 source data: `/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day`
  (see `/g/data/ob53/BARRA2/README.txt` for the dataset itself)

`barra_common.py` is shared plumbing (not run directly): it knows how to find/open
BARRA2 monthly files for a given variable, convert units, attach the correct
CRS for GeoTIFF export, build land-use/zone masks, and write colourised uint8
GeoTIFFs. The `VARIABLES` dict there defines which variables are supported
(`tasmax`, `sfcWindmax`, `uas`, `vas`) - add an entry to support a new one.

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

For each pixel/date in the mask where `sfcWindmax` exceeds a threshold,
records wind direction (from daily-mean `uas`/`vas`) and GRDC cropping
region.

```
python wind_exceedances_year.py --year 2022 --threshold 22
```

Writes `wind_exceedances_<year>.csv`. Run once per year, then combine:

```
python combine_exceedances.py --start-year 2017 --end-year 2025
```

Writes the combined CSV/GeoPackage plus a day-aggregated CSV (one row per
exceedance day, picking a representative pixel/coordinate per day).

### 3. Per-year masked wind summaries

Takes the seasonal max of a variable, restricted to the cropping mask;
writes a GeoTIFF and caches the masked pixel values for box-plotting.

```
python landuse_wind_year.py --variable sfcWindmax --year 2022 --start-month 9 --end-month 11
```

Then build a box plot across years from the cached `.npy` files:

```
python landuse_wind_boxplot.py --variable sfcWindmax --start-year 2017 --end-year 2025 --line 22
```

### 4. Australia-wide (unmasked) summaries

Seasonal/monthly max GeoTIFFs over the whole BARRA2 domain, no land-use
masking:

```
python make_monthly_max_geotiff.py --variable tasmax --year 2022 --month 9
python make_seasonal_max_geotiff.py --variable sfcWindmax --start-year 2017 --end-year 2025 --start-month 9 --end-month 11
```

### 5. Point/pixel time series

Daily time series (CSV + plot) of a variable at a given lat/lon:

```
python extract_point_timeseries.py --variable tasmax --lat -36.456671 --lon 148.262724 \
    --start-year 2022 --start-month 9 --end-year 2022 --end-month 11
```

Full-year time series at each year's worst exceedance pixel (from the
day-aggregated CSV produced in step 2):

```
python top_pixel_timeseries.py --by-day-csv wind_exceedances_2017-2025_by_day.csv
```

## Outputs

All scripts write to `/scratch/xe2/cb8590/wind-gusts2` by default
(override with `--out-dir`/`--out-path`). GeoTIFFs are uint8 with an
embedded GDAL colour table so they auto-colour when opened in QGIS.
