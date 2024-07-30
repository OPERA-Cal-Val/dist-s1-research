import backoff
import numpy as np
import rasterio
from mpire import WorkerPool
from pyproj import Transformer
from rasterio import RasterioIOError
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.windows import Window
from requests.exceptions import HTTPError


def get_utm_coords(url: str, lon: float, lat: float) -> tuple[float, float]:
    with rasterio.open(url) as src:

        src_crs = src.crs
    transformer = Transformer.from_crs(CRS.from_epsg(4326), src_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ConnectionError, RasterioIOError),
    max_tries=10,
    max_time=60,
    jitter=backoff.full_jitter,
)
def read_window_from_raster(
    url: str, utm_x: float, utm_y: float, window_size=3
) -> np.ndarray:
    with rasterio.open(url) as src:
        src_trans = src.transform

    if (window_size % 2) != 1:
        raise ValueError("window size must be odd")
    row, col = rowcol(src_trans, utm_x, utm_y)
    row_m1 = max(row - window_size // 2, 0)
    col_m1 = max(col - window_size // 2, 0)
    window = Window(col_m1, row_m1, window_size, window_size)

    with rasterio.open(url) as src:
        values = src.read(1, window=window)
    return values


@backoff.on_exception(
    backoff.expo, (RuntimeError, OSError), max_tries=20, max_time=60, jitter=backoff.full_jitter
)
def get_burst_time_series_around_point(
    urls: list[str], lon_site, lat_site, window_size=3
) -> list[np.ndarray]:
    utm_x, utm_y = get_utm_coords(urls[0], lon_site, lat_site)

    def read_window_from_raster_p(url: str):
        return read_window_from_raster(url, utm_x, utm_y, window_size=window_size)

    with WorkerPool(n_jobs=10, use_dill=True) as pool:
        vals_l = pool.map(
            read_window_from_raster_p,
            urls,
            progress_bar=True, progress_bar_style='std',
            concatenate_numpy_output=False,
        )
    return vals_l