from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window, transform as window_transform
from rasterio.transform import array_bounds, from_origin, rowcol
from rasterio.warp import transform_bounds
from rasterio import default_gtiff_profile
from dem_stitcher.rio_tools import reproject_profile_to_new_crs
from rasterio.crs import CRS


def get_row_col_from_profile(profile: dict, lon: float, lat: float) -> tuple[int, int]:
    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)

    x_utm, y_utm = transformer.transform(lon, lat)
    row, col = rowcol(profile["transform"], x_utm, y_utm)
    return int(row), int(col)


def get_window_around_lon_lat(tif: str | Path, lon: float, lat: float, buffer_pixels: int | None = None) -> Window:
    with rasterio.open(tif) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        x_utm, y_utm = transformer.transform(lon, lat)
        row, col = rowcol(src.transform, x_utm, y_utm, op=np.floor)
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            raise ValueError("Point is outside the bounds of the raster")
        if buffer_pixels is not None and buffer_pixels > 0:
            col = max(col - buffer_pixels, 0)
            row = max(row - buffer_pixels, 0)
            offset = 2 * buffer_pixels + 1
        elif buffer_pixels is None or buffer_pixels == 0:
            offset = 1
        window = Window(col, row, offset, offset)
    return window


def get_pixel_values_at_lon_lat(tif: str | Path, lon: float, lat: float, buffer_pixels: int | None = None):
    window = get_window_around_lon_lat(tif, lon, lat, buffer_pixels=buffer_pixels)
    with rasterio.open(tif) as src:
        val = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(
            transform=window_transform(window, src.transform),
            height=window.height,
            width=window.width,
        )
    return val, profile


def get_lat_lon_bounds_from_profile(profile: dict) -> tuple[float, float, float, float]:
    crs = profile["crs"]
    transform = profile["transform"]
    height = profile["height"]
    width = profile["width"]

    left, bottom, right, top = array_bounds(height, width, transform)

    min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)

    return min_lon, min_lat, max_lon, max_lat


def get_profile_from_extent(
    extent: tuple[float, float, float, float],
    height,
    width,
    src_crs: str,
    dtype: str = "float32",
    dst_crs: CRS | None = None,
    count: int = 1,
    nodata: float | int | None = None,
) -> dict:
    minx, miny, maxx, maxy = extent
    if minx > maxx or miny > maxy:
        raise ValueError("Invalid extent; minx > maxx or miny > maxy")
    x = np.linspace(minx, maxx, width)
    y = np.linspace(miny, maxy, height)
    resx = (x[-1] - x[0]) / width
    resy = (y[-1] - y[0]) / height
    resx = round(resx, 7)
    resy = round(resy, 7)
    if resx != resy:
        raise ValueError("x and y resolutions are not equal")
    transform = from_origin(x[0] - resx / 2, y[-1] + resy / 2, resx, resy)
    profile = default_gtiff_profile.copy()
    profile.update(
        dtype=dtype,
        count=count,
        height=height,
        width=width,
        crs=src_crs,
        transform=transform,
        nodata=nodata,
    )
    if dst_crs is not None:
        profile = reproject_profile_to_new_crs(profile, dst_crs)
    return profile
