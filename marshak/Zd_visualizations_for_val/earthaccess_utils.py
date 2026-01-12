import concurrent.futures
from warnings import warn
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from dem_stitcher.rio_window import get_window_from_extent
from rasterio.session import AWSSession
from rasterio.crs import CRS
from shapely.geometry import Polygon
from tqdm import tqdm

# CMR Concept ID for OPERA RTC-S1 from ASF
# https://cmr.earthdata.nasa.gov/search/concepts/C2777436413-ASF.html
OPERA_RTC_S1_CONCEPT_ID = "C2777436413-ASF"

OPERA_CONCEPT_IDS = {
    "RTC-S1": "C2777436413-ASF",
    "HLS": ["C2021957657-LPCLOUD", "C2021957295-LPCLOUD"],
    "DIST-HLS-ALERT": "C2746980408-LPCLOUD",
}


def earthaccess_opera_geo_search(
    bounding_box: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    collection: str,
) -> list[earthaccess.DataGranule]:
    earthaccess.login()
    if collection not in ["RTC-S1", "HLS", "DIST-HLS-ALERT"]:
        raise ValueError(f"Invalid collection: {collection}")
    else:
        concept_id = OPERA_CONCEPT_IDS[collection]

    results = earthaccess.search_data(
        concept_id=concept_id,
        bounding_box=bounding_box,
        temporal=(start_date, end_date),
    )
    return results


def get_granule_https_links(
    granules: list[earthaccess.DataGranule],
    layer_suffix: None | list[str] | str = None,
) -> list[str]:
    all_links = []
    for granule in granules:
        links = granule.data_links(access="external")
        if layer_suffix is not None:
            if isinstance(layer_suffix, list):
                links = []
                for suffix in layer_suffix:
                    assert isinstance(suffix, str), "suffixes in list is not str"
                    layer_pattern = f"{suffix}.tif"
                    links = [link for link in links if layer_pattern in link]
            elif isinstance(layer_suffix, str):
                layer_pattern = f"_{layer_suffix}.tif"
                links = [link for link in links if layer_pattern in link]
            else:
                TypeError(f"layer suffix has {type(layer_suffix)} and needs to be None | str | list[str]")
        all_links.extend(links)
    return all_links


def read_one_earthdata_url(
    url: str,
    extent: tuple[float, float, float, float] | None = None,
    extent_crs: str | CRS | int = 4326,
    auth_session: AWSSession | None = None,
) -> tuple[np.ndarray, dict]:
    if auth_session is None:
        auth_session = earthaccess.get_session()

    with rasterio.Env(
        GDAL_HTTP_COOKIEFILE="/tmp/cookies.txt",
        GDAL_HTTP_COOKIEJAR="/tmp/cookies.txt",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_COLUMN",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif",
    ):
        if isinstance(extent_crs, int):
            extent_crs = CRS.from_epsg(extent_crs)
        with rasterio.open(url) as dataset:
            prof = dataset.profile.copy()

            if extent is not None:
                window = get_window_from_extent(dataset.profile, extent, extent_crs)
                # Use masked=True if you expect NoData values
                arr = dataset.read(1, window=window)

                prof.update(
                    height=window.height,
                    width=window.width,
                    transform=rasterio.windows.transform(window, dataset.transform),
                    count=1,
                )
            else:
                arr = dataset.read(1)

    return arr, prof


def read_multiple_earthdata_urls(
    urls: list[str],
    extent: tuple[float, float, float, float] | None = None,
    extent_crs: str = "EPSG:4326",
    max_workers: int = 5,
) -> tuple[list[np.ndarray], list[dict]]:
    auth_session = earthaccess.get_requests_https_session()

    def read_with_session(url):
        return read_one_earthdata_url(url, auth_session=auth_session, extent=extent, extent_crs=extent_crs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data = list(tqdm(executor.map(read_with_session, urls), total=len(urls)))

    arrs, profiles = zip(*data)
    return list(arrs), list(profiles)


def convert_asf_url_to_cumulus(url: str) -> str:
    asf_base = "https://datapool.asf.alaska.edu/RTC/OPERA-S1/"
    cumulus_base = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/"

    if not (url.startswith(cumulus_base) or url.startswith(asf_base)):
        warn(f"URL {url} is not a valid ASF datapool or cumulus earthdatacloud URL.")
        return url

    if not url.startswith(asf_base):
        return url

    filename = url.split("/")[-1]
    granule_pol_parts = filename.rsplit("_", 1)
    if len(granule_pol_parts) != 2:
        raise ValueError(f"Could not extract granule name from filename: {filename}")

    granule_name = granule_pol_parts[0]
    new_url = f"{cumulus_base}{granule_name}/{filename}"
    return new_url


def flatten_earthaccess_result(
    result: dict,
    file_suffixes: list[str] | None = None,
    s3_urls: bool = False,
) -> dict:
    result_umm = result["umm"]
    flat = {}

    flat["granule_id"] = result_umm["GranuleUR"]

    temporal = result_umm["TemporalExtent"]["RangeDateTime"]
    flat["start_time"] = pd.Timestamp(temporal["BeginningDateTime"])
    flat["end_time"] = pd.Timestamp(temporal["EndingDateTime"])

    for attr in result_umm.get("AdditionalAttributes", []):
        name = attr["Name"].lower()
        values = attr["Values"]
        flat[name] = ",".join(values) if len(values) > 1 else values[0]

    if file_suffixes is None:
        file_suffixes = ["tif", "h5"]

    all_url_data = result_umm.get("RelatedUrls", [])

    if s3_urls:
        urls = [url["URL"] for url in all_url_data if url["URL"].startswith("s3://")]
    else:
        urls = [url["URL"] for url in all_url_data if url["URL"].startswith("https://")]
    filtered_urls = [url for url in urls if any(url.endswith(suffix) for suffix in file_suffixes)]

    for url in filtered_urls:
        stem = Path(url).stem
        suffix = Path(url).suffix

        if "_" in stem:
            separator = "_"
        else:
            separator = "."

        if ".tif" == suffix:
            parts = stem.split(separator)
            layer = parts[-1]
        elif ".h5" == suffix:
            layer = "h5"
        else:
            raise ValueError(f"Invalid file suffix: {stem}")
        if "asf" in url:
            url = convert_asf_url_to_cumulus(url)
        flat[f"url_{layer}".lower()] = url

    points = result_umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    coords = [(p["Longitude"], p["Latitude"]) for p in points]
    flat["geometry"] = Polygon(coords)

    return flat
