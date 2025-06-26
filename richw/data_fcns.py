#! /usr/bin/env python

import pandas as pd
import geopandas as gpd
import ast
import numpy as np
from rasterio import features

# Functions to access/manipulate OPERA data

def get_index(data: list[str], val: str) -> int:
    try:
        ind = data.index(val)
    except ValueError:
        ind = len(data)
    return ind

def get_first_change(labeled_ts):
    get_index_p = lambda val: get_index(labeled_ts, val)
    indices = list(map(get_index_p, ['VLmin', 'VLmaj', 'OCmin', 'OCmaj']))
    ind = min(indices)
    ind = ind if ind < len(labeled_ts) else -1
    return ind

def get_last_noChange(labeled_ts, change_ind):
    if change_ind == -1:
        return -1
    labeled_ts_r = (labeled_ts[:change_ind][::-1])
    ind_nc_r = get_index(labeled_ts_r, 'noChange')
    ind_nc = change_ind - ind_nc_r - 1
    return ind_nc

def get_first_change_from_row(row):
    change_label = row['overallLabel']
    if change_label in ['VLmin', 'VLmaj', 'OCmaj', 'OCmin']:
        dates_str = row.keys().tolist()[5:]
        dates = pd.to_datetime(dates_str)
        
        labeled_ts = list(row)[5:]
        ind_c = get_first_change(labeled_ts)
        change_date = dates[ind_c]
    else:
        change_date = pd.NaT
    return change_date

def get_last_obs_date_before_change(row):
    change_label = row['overallLabel']
    if change_label in ['VLmin', 'VLmaj', 'OCmaj', 'OCmin']:
        dates_str = row.keys().tolist()[5:]
        dates = pd.to_datetime(dates_str)
        
        labeled_ts = list(row)[5:]
        ind_c = get_first_change(labeled_ts)
        ind_nc = get_last_noChange(labeled_ts, ind_c)
        last_obs_date = dates[ind_nc] if ind_nc > -1 else pd.NaT
    else:
        last_obs_date = pd.NaT

    return last_obs_date

def indices(list, filtr=lambda x: bool(x)):
    return [i for i,x in enumerate(list) if filtr(x)]

def convert_to_list(v):
    if isinstance(v, str):
        v1 = ast.literal_eval(v.replace('nan','None'))
        v2 = [np.nan if x == None else x for x in v1]
        return v2
    return v


# Below from dist-s1-research/marshak/10_ad_hoc_data_generation/rasterize.py
def rasterize_shapes_to_array(shapes: list,
                              attributes: list,
                              profile: dict,
                              all_touched: bool = False,
                              dtype: str = np.float32) -> np.ndarray:
    """
    Takes a list of geometries and attributes to create an array. Roughly the
    inverse, in spirit, to `get_geopandas_features_from_array`.  For example,
    `shapes = df.geometry` and `attributes = df.label`, where df is a geopandas
    GeoDataFrame. We note the array is initialized as array of zeros.

    Parameters
    ----------
    shapes : list
        List of Shapely geometries.
    attributes : list
        List of attributes corresponding to shapes.
    profile : dict
        Rasterio profile in which shapes will be projected into, importantly
        the transform and dimensions specified.
    all_touched : bool
        Whether factionally covered pixels are written with specific value or
        ignored. See `rasterio.features.rasterize`.
    dtype : str
        The initial array is np.zeros and dtype can be specified as a numpy
        dtype or appropriate string.

    Returns
    -------
    np.ndarray:
        The output array determined with profile.
    """
    out_arr = np.zeros((profile['height'], profile['width']), dtype=dtype)

    # this is where we create a generator of geom, value pairs to use in
    # rasterizing
    shapes = [(geom, value) for geom, value in zip(shapes, attributes)]
    burned = features.rasterize(shapes=shapes,
                                out=out_arr,
                                transform=profile['transform'],
                                all_touched=all_touched)

    return burned
