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

# Per-variable settings needed to turn raw model output into a sensible,
# colourised uint8 GeoTIFF. Add a new entry here to support a new variable.
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


def open_variable(variable, start_year, start_month, end_year=None, end_month=None, chunks=None):
    """Open `variable` across the given month range, with units converted per
    VARIABLES[variable]['convert']. Loads eagerly (chunks=None) by default -
    this node has a single CPU, so dask's 'auto' chunking just adds overhead
    and was observed to destabilise memory use; pass chunks="auto" to opt
    back into a dask-backed lazy array if needed."""
    files = find_files(variable, start_year, start_month, end_year, end_month)
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    da = ds[variable]
    cfg = VARIABLES[variable]
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
