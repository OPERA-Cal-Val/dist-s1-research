import contextily as cx
from rasterio.crs import CRS

from rio_utils import get_profile_from_extent


def get_basemap_for_matplotlib(w, s, e, n, zoom="auto", source=cx.providers.Esri.WorldImagery):
    # Google Satellite: https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}
    img, extent = cx.bounds2img(w, s, e, n, zoom=zoom, source=source, ll=True)
    return img, extent


def get_basemap_for_rasterio(
    w,
    s,
    e,
    n,
    zoom="auto",
    source=cx.providers.Esri.WorldImagery,
    nodata: float | int | None = None,
):
    img, extent = get_basemap_for_matplotlib(w, s, e, n, zoom=zoom, source=source)
    minx, maxx, miny, maxy = extent
    extent = (minx, miny, maxx, maxy)
    h, w, c = img.shape
    profile = get_profile_from_extent(extent, h, w, CRS.from_epsg(3857), count=c, nodata=nodata)
    img_rasterio = img.transpose(2, 0, 1)
    return img_rasterio, profile
