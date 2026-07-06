"""Shared helpers for working with BARRA2 AUST-04 daily netCDF data.

Generalised across variables so the same plumbing can be reused for e.g.
tasmax (daily max temperature) and sfcWindmax (daily max wind speed) -
just point at a different key in VARIABLES.
"""
import os

# This node has no outbound internet access. The conda env activation already
# sets PROJ_NETWORK=ON, so with PROJ's network grid lookups left on, any
# reprojection between two different datums (e.g. GDA94 -> WGS84 for the
# NLUM/GRDC inputs) makes pyproj try to fetch a grid from the PROJ CDN, hang
# for a long time, and eventually get killed. Force it off (not setdefault -
# the env already has it set to "ON") before anything touches pyproj.
os.environ["PROJ_NETWORK"] = "OFF"

from pathlib import Path

import numpy as np
import pyproj
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rasterio.crs import CRS

pyproj.network.set_network_enabled(False)

BARRA_DAY_ROOT = Path(
    "/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/day"
)
FILENAME_TEMPLATE = "{variable}_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1_day_{ym}-{ym}.nc"

# The BARRA2/CORDEX grid is geographic (regular lat/lon) but technically
# defined on a sphere, not the WGS84 ellipsoid: the `crs` grid_mapping
# variable in the netCDFs has grid_mapping_name=latitude_longitude,
# earth_radius=6371229. We used to encode that exact sphere via a custom
# proj4 string, but it has no EPSG code, and GIS tools (QGIS in particular)
# could not reliably identify/reproject it - rasters appeared correctly
# georeferenced in gdalinfo but failed to line up with any basemap in QGIS.
# The positional error from instead labelling these coordinates as WGS84
# (EPSG:4326) is well under a metre - negligible next to the ~4km pixels -
# so we use the standard, universally-recognised CRS for all GIS output.
BARRA_CRS = CRS.from_epsg(4326)

# --- Wind-gust adjustment (applied to wsgsmax) -----------------------------
# wsgsmax is BARRA-C2's modelled wind gust (standard_name wind_speed_of_gust):
# the daily max of the 3-s peak gust at 10 m. It is the right variable for a
# damaging-gust analysis - sfcWindmax is only the daily max of the *hourly-mean*
# wind, which is what Pinera-Chavez (2016) had to gustify by hand. We do NOT
# redo their empirical mean->gust conversion; instead we take the model gust and
# apply two corrections so it is comparable to their 22 m/s lodging threshold:
#
# 1. Bias correction. Bush et al. (2025) / BoM (2025) report a known positive
#    bias in the BARRA2 gust diagnostic from a mis-set constant in the gust
#    parameterisation; the corrected constant reduces the simulated gust by ~10%.
GUST_BIAS_FACTOR = 0.90
#
# 2. Height correction 10 m -> 2 m. wsgsmax is a 10 m gust; the Pinera-Chavez
#    lodging thresholds are gusts at the ~2 m "pertinent height for crop lodging"
#    (Baker et al. 1998). We scale with the neutral-log wind profile over a wheat
#    canopy, reusing the roughness parameters of Berry et al. (2003b) /
#    Pinera-Chavez (2016, Eq. 7, p. 328): for a crop of height h = 1 m, displacement height
#    d = 0.75 h and roughness length z0 = (h - d)/3 (= 1/12 m). The factor is
#    ln((2 - d)/z0) / ln((10 - d)/z0) ~= 0.575. (Strictly this profile describes
#    the mean wind - Pinera applied it to means before gustifying - so applying
#    it to the model gust is an approximation; set the height constants equal to
#    disable.) Set either factor to 1.0 to turn a correction off.
_CROP_HEIGHT_M = 1.0
_DISPLACEMENT_M = 0.75 * _CROP_HEIGHT_M
_ROUGHNESS_M = (_CROP_HEIGHT_M - _DISPLACEMENT_M) / 3.0
_GUST_REF_HEIGHT_M = 10.0  # wsgsmax native measurement height
_CROP_WIND_HEIGHT_M = 2.0  # pertinent height for crop lodging
GUST_HEIGHT_FACTOR = float(
    np.log((_CROP_WIND_HEIGHT_M - _DISPLACEMENT_M) / _ROUGHNESS_M)
    / np.log((_GUST_REF_HEIGHT_M - _DISPLACEMENT_M) / _ROUGHNESS_M)
)
# Combined multiplier applied to raw wsgsmax (~0.518).
GUST_ADJUSTMENT = GUST_BIAS_FACTOR * GUST_HEIGHT_FACTOR

# --- Pinera-Chavez (2016) empirical mean->gust conversion (for comparison) --
# An ALTERNATIVE to the model gust: reconstruct a gust from the daily-max
# hourly-MEAN wind (sfcWindmax) the way Pinera-Chavez (2016) had to, because
# they lacked gust observations. This is NOT used for the main analysis (which
# uses the model gust wsgsmax); it exists so the two gust estimates can be
# plotted against each other. Their Eq. 6 (Pinera-Chavez et al. 2016, p. 328;
# after Berry et al. 2003b) turns an hourly-mean wind Um into a peak gust of
# duration tau:
#     Ugust = Um * [1 + 0.42 * TI * ln(T_ref / tau)]
# with turbulence intensity TI = sigma/Um = 0.5 (Finnigan 1979, wind over a
# wheat crop), reference averaging period T_ref = 3600 s (1 h, matching the
# hourly mean) and gust duration tau = 0.3 s. The bracket is a constant gust
# factor (~2.97). We gustify the mean at its native 10 m first - this is the
# same measurement height as the model's own raw wsgsmax, so the two are
# directly comparable at that intermediate stage - and only then bring the
# result down to 2 m crop height with the same GUST_HEIGHT_FACTOR used for
# the model gust above, so pinera_gust = sfcWindmax * PINERA_GUST_FACTOR *
# GUST_HEIGHT_FACTOR (~1.71 overall). NB this assumes very high near-canopy
# turbulence and so gives a much larger 2 m gust than log-lawing the model's
# 10 m gust down to 2 m does - that divergence is the point of the comparison.
PINERA_TURBULENCE_INTENSITY = 0.5
PINERA_AVERAGING_PERIOD_S = 3600.0
PINERA_GUST_DURATION_S = 0.3
PINERA_GUST_FACTOR = float(
    1.0 + 0.42 * PINERA_TURBULENCE_INTENSITY * np.log(PINERA_AVERAGING_PERIOD_S / PINERA_GUST_DURATION_S)
)


# Per-variable settings needed to turn raw model output into a sensible,
# colourised uint8 GeoTIFF. Add a new entry here to support a new variable.
# Keys are *logical* names selected via --variable. Most map 1:1 to an on-disk
# BARRA variable; a few are DERIVED - they set 'source' to the on-disk variable
# to read and use 'convert' to transform it (see open_variable). The
# sfcWindmax/wsgsmax family below exists so the mean wind, the model gust, and
# the Pinera-Chavez empirical gust can all be compared (see compare_point_gusts.py).
VARIABLES = {
    "tasmax": dict(
        long_name="Daily max temperature",
        units_out="degC",
        convert=lambda da: da - 273.15,  # K -> degC
        cmap="RdYlBu_r",
    ),
    "sfcWindmax": dict(
        long_name="Daily max wind speed",
        units_out="m s-1",
        convert=lambda da: da,  # already m/s
        cmap="viridis",
    ),
    "pr": dict(
        long_name="Daily total rainfall",
        units_out="mm",
        convert=lambda da: da * 86400.0,  # kg m-2 s-1 (mean) -> mm/day
        cmap="Blues",
    ),
    "wsgsmax": dict(
        long_name="Daily max wind gust (bias- & height-adjusted to 2 m)",
        units_out="m s-1",
        # Raw model gust (m/s) scaled by the ~10% bias correction and the
        # 10 m -> 2 m crop-height factor documented above.
        convert=lambda da: da * GUST_ADJUSTMENT,
        cmap="viridis",
    ),
    # --- Comparison-only derived variables (not used by the main pipeline) ---
    "wsgsmax_raw": dict(
        long_name="Daily max wind gust (raw model, 10 m)",
        units_out="m s-1",
        source="wsgsmax",  # same file as wsgsmax, but no bias/height adjustment
        convert=lambda da: da,
        cmap="viridis",
    ),
    "wsgsmax_bias": dict(
        long_name="Daily max wind gust (bias-adjusted only, still at 10 m)",
        units_out="m s-1",
        source="wsgsmax",  # same file as wsgsmax, but no height adjustment
        convert=lambda da: da * GUST_BIAS_FACTOR,
        cmap="viridis",
    ),
    "sfcWindmax_2m": dict(
        long_name="Daily max wind speed (mean, height-corrected to 2 m)",
        units_out="m s-1",
        source="sfcWindmax",
        convert=lambda da: da * GUST_HEIGHT_FACTOR,
        cmap="viridis",
    ),
    "pinera_gust": dict(
        long_name="Daily max gust (Pinera-Chavez conversion from mean wind, 2 m)",
        units_out="m s-1",
        source="sfcWindmax",  # empirical mean->gust: gust factor (10 m) -> 2 m
        convert=lambda da: da * PINERA_GUST_FACTOR * GUST_HEIGHT_FACTOR,
        cmap="viridis",
    ),
    "uas": dict(
        long_name="Daily mean eastward wind",
        units_out="m s-1",
        convert=lambda da: da,  # already m/s
        cmap="RdBu_r",
    ),
    "vas": dict(
        long_name="Daily mean northward wind",
        units_out="m s-1",
        convert=lambda da: da,  # already m/s
        cmap="RdBu_r",
    ),
}


def _month_range(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def find_files(variable, start_year, start_month, end_year=None, end_month=None):
    """Return sorted monthly BARRA2 file paths for `variable` covering the given
    inclusive year/month range (single month if end_year/end_month omitted)."""
    if end_year is None:
        end_year, end_month = start_year, start_month
    var_dir = BARRA_DAY_ROOT / variable / "latest"
    files = []
    for y, m in _month_range(start_year, start_month, end_year, end_month):
        ym = f"{y}{m:02d}"
        path = var_dir / FILENAME_TEMPLATE.format(variable=variable, ym=ym)
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(str(path))
    return files


def available_month_range(variable):
    """Return (start_year, start_month, end_year, end_month) spanning every
    monthly file present on disk for `variable`, by scanning filenames
    rather than hardcoding a range that will go stale as new months land."""
    var_dir = BARRA_DAY_ROOT / variable / "latest"
    yms = sorted(p.stem.rsplit("_", 1)[-1].split("-")[0] for p in var_dir.glob(f"{variable}_*.nc"))
    if not yms:
        raise FileNotFoundError(f"No files found for variable {variable!r} in {var_dir}")
    first, last = yms[0], yms[-1]
    return int(first[:4]), int(first[4:6]), int(last[:4]), int(last[4:6])


def open_variable(variable, start_year, start_month, end_year=None, end_month=None, chunks=None):
    """Open `variable` across the given month range, with units converted per
    VARIABLES[variable]['convert']. Loads eagerly (chunks=None) by default -
    this node has a single CPU, so dask's 'auto' chunking just adds overhead
    and was observed to destabilise memory use; pass chunks="auto" to opt
    back into a dask-backed lazy array if needed.

    `variable` is a *logical* name (a key in VARIABLES), which may be a derived
    quantity: if its entry sets 'source', that on-disk BARRA variable is read
    instead and 'convert' turns it into the derived quantity (e.g. pinera_gust
    reads sfcWindmax then applies the empirical mean->gust factor)."""
    cfg = VARIABLES[variable]
    source = cfg.get("source", variable)  # on-disk BARRA variable to read
    files = find_files(source, start_year, start_month, end_year, end_month)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    da = ds[source]
    da = cfg["convert"](da)
    da.attrs["units"] = cfg["units_out"]
    da.attrs["long_name"] = cfg["long_name"]
    return da


def prepare_spatial(da):
    """Orient north-up (descending lat) and attach the BARRA2 CRS so rioxarray
    can write a correctly georeferenced GeoTIFF."""
    da = da.sortby("lat", ascending=False)
    da = da.rio.write_crs(BARRA_CRS)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    return da


def to_uint8_with_colormap(da2d, vmin=None, vmax=None, cmap_name="viridis", nodata_value=0):
    """Scale a 2D float DataArray into uint8 (1-255, 0 reserved for nodata)
    and build a matching GDAL-style colormap dict {index: (r, g, b, a)}.
    vmin/vmax default to the DataArray's own min/max when not given."""
    import matplotlib.cm as cm

    arr = np.asarray(da2d.values, dtype="float64")
    valid = np.isfinite(arr)

    if vmin is None:
        vmin = float(arr[valid].min())
    if vmax is None:
        vmax = float(arr[valid].max())

    clipped = np.where(valid, np.clip(arr, vmin, vmax), vmin)  # NaNs replaced before the uint8 cast, overwritten below
    norm = (clipped - vmin) / (vmax - vmin)
    scaled = np.round(norm * 254 + 1).astype(np.uint8)
    scaled[~valid] = nodata_value

    out = da2d.copy(data=scaled)
    out = out.rio.write_nodata(nodata_value)

    cmap = cm.get_cmap(cmap_name)
    colormap = {0: (0, 0, 0, 0)}  # transparent nodata
    for i in range(1, 256):
        r, g, b, a = cmap((i - 1) / 254)
        colormap[i] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255)
    return out, colormap, vmin, vmax


def to_uint8_direct(da2d, cmap_name="viridis", nodata_value=0):
    """Round a 2D float DataArray straight to uint8 - pixel value N literally
    means N units (e.g. N m/s), no min/max rescaling. Only suitable for
    variables that stay within ~1-255 in their native units (wind speed in
    m/s, not temperature in degC). Values below 1 are clamped up to 1 since
    0 is reserved for nodata - fine for wind maxima, which never reach 0.
    The colour table is still stretched across the data's own min/max for
    visual contrast, even though the pixel values themselves are untouched."""
    import matplotlib.cm as cm

    arr = np.asarray(da2d.values, dtype="float64")
    valid = np.isfinite(arr)

    scaled = np.clip(np.round(arr), 1, 255).astype(np.uint8)
    scaled[~valid] = nodata_value

    out = da2d.copy(data=scaled)
    out = out.rio.write_nodata(nodata_value)

    vmin = float(arr[valid].min())
    vmax = float(arr[valid].max())

    cmap = cm.get_cmap(cmap_name)
    colormap = {0: (0, 0, 0, 0)}  # transparent nodata
    for i in range(1, 256):
        norm = 0.0 if vmax <= vmin else min(max((i - vmin) / (vmax - vmin), 0.0), 1.0)
        r, g, b, a = cmap(norm)
        colormap[i] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255)
    return out, colormap, vmin, vmax


def load_landuse_mask(tif_path, reference_da, raw_threshold, resampling=None):
    """Resample a probability/coverage raster (e.g. an NLUM probSurf tif, on
    its own raw value scale) onto `reference_da`'s grid and threshold it.

    Returns a boolean DataArray that is True where the area-weighted average
    of the source raster exceeds `raw_threshold`. Deliberately rebuilt as a
    copy of `reference_da` (rather than returned straight from
    reproject_match): reproject_match's output keeps dims y/x with
    coordinates regenerated from the affine transform, which are close to
    but not bit-identical to reference_da's lat/lon - combining the two as
    labelled DataArrays later makes xarray treat them as unrelated
    dimensions and broadcast a full outer product instead of aligning
    elementwise (this previously caused an OOM-inducing 1018x1298x1018x1298
    array). Sharing reference_da's exact dims/coords up front avoids that.
    """
    from rasterio.enums import Resampling

    if resampling is None:
        resampling = Resampling.average

    surface = rioxarray.open_rasterio(tif_path, masked=True).squeeze("band", drop=True)
    resampled = surface.rio.reproject_match(reference_da, resampling=resampling)
    matched = np.nan_to_num(resampled.values, nan=-np.inf) > raw_threshold
    return reference_da.copy(data=matched)


def load_zone_mask(gpkg_path, reference_da):
    """Rasterize every polygon in a vector file (e.g. GRDC cropping zones)
    onto `reference_da`'s grid, returning a boolean DataArray that is True
    for pixels whose centre falls inside at least one polygon."""
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(gpkg_path)
    if reference_da.rio.crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(reference_da.rio.crs)

    burned = rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None),
        out_shape=reference_da.shape,
        transform=reference_da.rio.transform(),
        fill=0,
        dtype="uint8",
    )
    return reference_da.copy(data=(burned == 1))


def load_zone_labels(gpkg_path, reference_da, label_col):
    """Rasterize every polygon in a vector file onto `reference_da`'s grid,
    burning in a 1-based zone index per polygon (0 = outside any zone).

    Returns (zone_id_grid, id_to_name) where zone_id_grid is a 2D uint16
    numpy array aligned to reference_da, and id_to_name maps each non-zero
    id back to the polygon's `label_col` attribute value.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(gpkg_path).reset_index(drop=True)
    if reference_da.rio.crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(reference_da.rio.crs)

    shapes = ((geom, idx + 1) for idx, geom in enumerate(gdf.geometry) if geom is not None)
    zone_id_grid = rasterize(
        shapes,
        out_shape=reference_da.shape,
        transform=reference_da.rio.transform(),
        fill=0,
        dtype="uint16",
    )
    id_to_name = {idx + 1: name for idx, name in enumerate(gdf[label_col])}
    return zone_id_grid, id_to_name


def wind_uv_to_direction(u, v):
    """Convert eastward/northward wind components (u, v, in any consistent
    units) to meteorological direction: degrees clockwise from north that
    the wind is blowing FROM, plus the nearest 8-point compass label.
    Works elementwise on scalars or numpy arrays."""
    deg = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    labels = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    idx = np.round(deg / 45.0).astype(int) % 8
    compass = labels[idx]
    return deg, compass


def write_geotiff_with_colormap(da_uint8, colormap, out_path):
    """Write a uint8 DataArray to GeoTIFF and embed a GDAL colour table so
    QGIS auto-applies colours on open."""
    import rasterio

    da_uint8.rio.to_raster(out_path, dtype="uint8")
    with rasterio.open(out_path, "r+") as dst:
        dst.write_colormap(1, colormap)
