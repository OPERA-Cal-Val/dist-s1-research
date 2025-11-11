from tile_mate.stitcher import get_all_tile_data
import pandas as pd
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
from affine import Affine
from rasterio.features import shapes
import multiprocessing as mp

def get_geopandas_features_from_array(arr: np.ndarray,
                                      transform: Affine,
                                      label_name: str = 'label',
                                      mask: np.ndarray = None,
                                      connectivity: int = 4) -> list:
    # see rasterio.features.shapes - needs all false values to be no data areas
    if mask is None:
        mask = np.zeros(arr.shape, dtype=bool)
    feature_list = list(shapes(arr,
                               mask=~mask,
                               transform=transform,
                               connectivity=connectivity))
    geo_features = list({'properties': {label_name: int(value)},
                         'geometry': geometry}
                        for i, (geometry, value) in enumerate(feature_list))
    return geo_features


def vectorize_one_tile(url):
    with rasterio.open(url) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    mask = (data <= 2).astype(np.uint8)
    if mask.sum():
        features = get_geopandas_features_from_array(mask, 
                                                     transform, 
                                                     mask=~(mask.astype(bool)
                                                     )
                                                )
        df = gpd.GeoDataFrame.from_features(features, 
                                            crs=crs)
    else:
        df = gpd.GeoDataFrame(geometry=[], crs=crs)
    return df

def vectorize_all_tiles(urls, output_path, n_workers = 5):
    all_gdfs = []
    first_crs = None
    with mp.Pool(n_workers) as p:
        all_gdfs = list(tqdm(p.imap(vectorize_one_tile, urls), total=len(urls)))
    non_empty_gdfs = [df for df in all_gdfs if not df.empty]
    print("Individual tile processing complete. Combining results...")
    combined_gdf = pd.concat(non_empty_gdfs, ignore_index=True)
    return combined_gdf