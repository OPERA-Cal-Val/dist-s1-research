
import sys
import os
import inspect
from pathlib import Path
import contextily as ctx
from datetime import datetime, timedelta
import requests
from typing import Iterable, List

import math
from shapely.geometry import Point
from shapely.geometry import box
from geopy.distance import geodesic
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from PIL import Image
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches
import leafmap

import dist_s1
from dist_s1.packaging import (
  convert_geotiff_to_png
)
from dist_s1.dist_processing import (
    merge_burst_metrics_and_serialize,
)
from dist_s1.workflows import(
    run_dist_s1_workflow,
    run_dist_s1_sas_prep_workflow,
    run_dist_s1_sas_workflow,
)
from dist_s1.constants import (
    BASE_DATE_FOR_CONFIRMATION,
    DISTLABEL2VAL,
    DIST_STATUS_CMAP,
    TIF_LAYERS,
    TIF_LAYER_DTYPES,
    TIF_LAYER_NODATA_VALUES,
)
from dist_s1_enumerator import (
    enumerate_one_dist_s1_product,
    get_burst_ids_in_mgrs_tiles,
    get_burst_table_from_mgrs_tiles,
    get_lut_by_mgrs_tile_ids,
    get_mgrs_tiles_overlapping_geometry,
    get_rtc_s1_ts_metadata_from_mgrs_tiles,
    get_mgrs_table,
)
from dist_s1_enumerator.mgrs_burst_data import (
    get_mgrs_tile_table_by_ids,
    get_burst_ids_in_mgrs_tiles
)
from dist_s1_enumerator.dist_enum import enumerate_dist_s1_products
from dist_s1.data_models.defaults import (
    DEFAULT_APPLY_DESPECKLING,
    DEFAULT_APPLY_LOGIT_TO_INPUTS,
    DEFAULT_APPLY_WATER_MASK,
    DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_CONFIDENCE_UPPER_LIM,
    DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    DEFAULT_DEVICE,
    DEFAULT_DST_DIR,
    DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_INTERPOLATION_METHOD,
    DEFAULT_LOOKBACK_STRATEGY,
    DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
    DEFAULT_MAX_OBS_NUM_YEAR,
    DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    DEFAULT_MEMORY_STRATEGY,
    DEFAULT_METRIC_VALUE_UPPER_LIM,
    DEFAULT_MODEL_CFG_PATH,
    DEFAULT_MODEL_COMPILATION,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_WTS_PATH,
    DEFAULT_NO_COUNT_RESET_THRESH,
    DEFAULT_NO_DAY_LIMIT,
    DEFAULT_N_ANNIVERSARIES_FOR_MW,
    DEFAULT_N_WORKERS_FOR_DESPECKLING,
    DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_PERCENT_RESET_THRESH,
    DEFAULT_POST_DATE_BUFFER_DAYS,
    DEFAULT_PRIOR_DIST_S1_PRODUCT,
    DEFAULT_PRODUCT_DST_DIR,
    DEFAULT_SRC_WATER_MASK_PATH,
    DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    DEFAULT_TQDM_ENABLED,
    DEFAULT_USE_DATE_ENCODING,
)

from dist_s1.data_models.runconfig_model import RunConfigData

import plot_fcns
import debug_defs


def show_mgrs(target_name,i_mgrs = 0):
    geo_loc_file = target_name + '.geojson'
    plot_mgrs_png = 'map_mgrs_' + target_name + '.png'
    plot_bursts_png = 'map_bursts_' + target_name + '.png'

    gdf = gpd.read_file(geo_loc_file)
    exploded_gdf = gdf.explode(index_parts=True)
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = box(minx, miny, maxx, maxy) 
    geom = exploded_gdf.geometry.iloc[0]
    df_mgrs = get_mgrs_table()
    intersect_mask = df_mgrs.intersects(geom)
    intersect_df = df_mgrs[intersect_mask]
    df_mgrs_tiles = get_mgrs_tiles_overlapping_geometry(geom)

    do_plots = True

    if do_plots:
        fig, ax = plt.subplots()
        df_mgrs_tiles_plot = df_mgrs_tiles.copy()
        df_mgrs_tiles_plot.geometry = df_mgrs_tiles_plot.geometry.boundary

        df_mgrs_tiles_plot.plot(
            column="mgrs_tile_id", categorical=True, legend=True, ax=ax
        )

        exploded_gdf.boundary.plot(
            ax=ax, edgecolor="red", linewidth=2 )

        ctx.add_basemap(
            ax, crs=df_mgrs_tiles.crs.to_string(),
             source=ctx.providers.CartoDB.Voyager
        )
        fig.savefig(plot_mgrs_png,dpi=300,bbox_inches="tight")


    mgrs_tile_id = df_mgrs_tiles['mgrs_tile_id'][i_mgrs]
    df_mgrs1 = get_mgrs_tile_table_by_ids(mgrs_tile_id)
    df_bursts = get_burst_table_from_mgrs_tiles([mgrs_tile_id])

    if do_plots:
        plotname = 'map_' + mgrs_tile_id + '.png'
        plot_fcns.plot_mgrs_bursts(plotname,df_mgrs1,None,bbox,df_bursts)

    return mgrs_tile_id

def show_mgrs1(gdf,id,i_mgrs):
    plot_mgrs_png = 'map_mgrs_' + id + '.png'
    plot_bursts_png = 'map_bursts_' + id + '.png'

    exploded_gdf = gdf.explode(index_parts=True)
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = box(minx, miny, maxx, maxy) 
    geom = exploded_gdf.geometry.iloc[0]
    df_mgrs = get_mgrs_table()
    intersect_mask = df_mgrs.intersects(geom)
    intersect_df = df_mgrs[intersect_mask]
    df_mgrs_tiles = get_mgrs_tiles_overlapping_geometry(geom)

    do_plots = True

    if do_plots:
        fig, ax = plt.subplots()
        df_mgrs_tiles_plot = df_mgrs_tiles.copy()
        df_mgrs_tiles_plot.geometry = df_mgrs_tiles_plot.geometry.boundary

        df_mgrs_tiles_plot.plot(
            column="mgrs_tile_id", categorical=True, legend=True, ax=ax
        )

        exploded_gdf.boundary.plot(
            ax=ax, edgecolor="red", linewidth=2 )

        ctx.add_basemap(
            ax, crs=df_mgrs_tiles.crs.to_string(),
             source=ctx.providers.CartoDB.Voyager
        )
        fig.savefig(plot_mgrs_png,dpi=300,bbox_inches="tight")


    mgrs_tile_id = df_mgrs_tiles['mgrs_tile_id'][i_mgrs]
    df_mgrs1 = get_mgrs_tile_table_by_ids(mgrs_tile_id)
    df_bursts = get_burst_table_from_mgrs_tiles([mgrs_tile_id])

    if do_plots:
        plotname = 'map_' + mgrs_tile_id + '.png'
        plot_fcns.plot_mgrs_bursts(plotname,df_mgrs1,None,bbox,df_bursts)

    return mgrs_tile_id

def enumerate_post_dates(mgrs_tile_id,start_dt,stop_dt):
    # Enumerate all possible post dates
    df_ts = get_rtc_s1_ts_metadata_from_mgrs_tiles(mgrs_tile_id)
    df_products = enumerate_dist_s1_products(df_ts, [mgrs_tile_id])
    post_ind = df_products.input_category == "post"
    df_prod_post = df_products[post_ind]
    # Select post dates within user window
    df_sel_prod_post = df_prod_post[
        (df_prod_post['acq_dt'].dt.tz_localize(None) >= start_dt) &
        (df_prod_post['acq_dt'].dt.tz_localize(None) <= stop_dt)
    ]
    
    post_dates = df_sel_prod_post.acq_date_for_mgrs_pass.unique()
    sorted_post_dates = np.sort(post_dates.astype('datetime64'))

    return sorted_post_dates,df_sel_prod_post

def run_mgrs_seq_local(mgrs_tile,sorted_post_dates,
    df_sel_prod_post,sas_prep_only,dir_prefix='',subset_bbox=None):
    # Run Dist-S1 processing with confirmation on the supplied
    # post-date sequence
    dst_dir = dir_prefix + '/intermed'
    input_data_dir = dir_prefix + '/'
    product_dst_dir = dir_prefix + '/'
    apply_water_mask = 'false'
    algo_config_path = dir_prefix + './alg_config_baseline.yml'
    prior = None
    prod_names = []
    mgrs_tiles = []
    post_dates = []
    prod_dates = []
    for post_date in sorted_post_dates:
        str_post_date = str(post_date)
        df_prod_date1 = df_sel_prod_post[
            df_sel_prod_post.acq_date_for_mgrs_pass == str_post_date
            ].reset_index(drop=True)
        #track1 = df_prod_date1.track_number[0].item()
        acq_groups = df_prod_date1.acq_group_id_within_mgrs_tile.unique()
        for acq_group in acq_groups:
          df_prod_date2 = df_prod_date1[df_prod_date1.acq_group_id_within_mgrs_tile == acq_group]
          track1 = list(df_prod_date2['track_number'])[0]
          acq_group1 = list(df_prod_date2['acq_group_id_within_mgrs_tile'])[0]
          all_match = (df_prod_date2['acq_group_id_within_mgrs_tile'] == 
            acq_group1).all()
          if not all_match:
            raise Exception('acq_group mismatch')
          if sas_prep_only:
            rc1 = run_dist_s1_sas_prep_workflow(mgrs_tile,str_post_date,track1,
                dst_dir=dst_dir,
                input_data_dir=input_data_dir,
                product_dst_dir=product_dst_dir,
                apply_water_mask=apply_water_mask,
                prior_dist_s1_product=prior,
                algo_config_path=algo_config_path)
          elif subset_bbox is not None:
            rc1,gdf1,width,hgt = run_dist_s1_workflow_subset(
                mgrs_tile,str_post_date,track1,
                subset_bbox=subset_bbox,
                dst_dir=dst_dir,
                input_data_dir=input_data_dir,
                product_dst_dir=product_dst_dir,
                apply_water_mask=apply_water_mask,
                prior_dist_s1_product=prior,
                algo_config_path=algo_config_path)
          else:
            rc1 = run_dist_s1_workflow(mgrs_tile,str_post_date,track1,
                dst_dir=dst_dir,
                input_data_dir=input_data_dir,
                product_dst_dir=product_dst_dir,
                apply_water_mask=apply_water_mask,
                prior_dist_s1_product=prior,
                algo_config_path=algo_config_path)

          nominal_prior = rc1.product_dst_dir / rc1.product_name
          if nominal_prior.is_dir():
              # Previous product was created so ok to move prior forward.
              # Note that subsetting the outputs has to wait until the
              # confirmation process is finished using the full size files
              # Alternatively could separate confirmation out of the
              # sas_workflow, but this seems easier for now.
              if subset_bbox is not None:
                  print('Subsetting output product files in prior product')
                  subset_outputs(prior,gdf1,width,hgt)
              prior = nominal_prior

          prod_names.append(str(prior))
          parts = str(rc1.product_name).split('_')
          mgrs_tile_str = parts[3]
          post_date_str = parts[4]
          prod_date_str = parts[5]
          mgrs_tiles.append(mgrs_tile_str)
          post_dates.append(post_date_str)
          prod_dates.append(prod_date_str)

    df_prod = pd.DataFrame({
        'prod_name': prod_names,
        'mgrs_tile': mgrs_tiles,
        'post_date': post_dates,
        'prod_date': prod_dates
    })
    df_prod.to_parquet('df_prod.parquet')

    if subset_bbox is not None:
        # Last output date still needs subsetting of products
        print('Subsetting final output product set')
        subset_outputs(prior,gdf1,width,hgt)

    return df_prod

def subset_outputs(out_dir,gdf1,width,hgt):
    if out_dir is not None and out_dir.is_dir():
        for out_path in out_dir.glob("*.tif"):
            with rasterio.open(out_path) as out:
                # Form subset output product using centroids,width,hgt
                subset_profile,subset_out = subset_geotif(gdf1,width,hgt,out)
            with rasterio.open(out_path,'w', **subset_profile) as subset:
                # Over-write output tif's with subset data
                subset.write(subset_out,1)

def subset_list(in_paths,out_paths,gdf1,width,hgt):
    if (in_paths is not None
      and len(in_paths) > 0
      and out_paths is not None
      and len(out_paths) > 0):
        for in_path,out_path in zip(in_paths,out_paths):
            with rasterio.open(in_path) as src:
                # Form subset output product using centroids,width,hgt
                subset_profile,subset_out = subset_geotif(gdf1,width,hgt,src)
            with rasterio.open(out_path,'w', **subset_profile) as subset:
                # write output tif's with subset data
                subset.write(subset_out,1)
    return out_paths

def prod_from_dir(prod_dir,prod_base="OPERA_L3_DIST-ALERT-S1"):
    basedir = Path(prod_dir)
    prod_names = []
    mgrs_tiles = []
    post_dates = []
    prod_dates = []
    for dir_path in basedir.glob(f"{prod_base}*"):
        if dir_path.is_dir():
            prod_names.append(dir_path)
            parts = str(dir_path).split('_')
            mgrs_tile_str = parts[3][1:]
            post_date_str = parts[4]
            prod_date_str = parts[5]
            mgrs_tiles.append(mgrs_tile_str)
            post_dates.append(post_date_str)
            prod_dates.append(prod_date_str)

    df_prod = pd.DataFrame({
        'prod_name': prod_names,
        'mgrs_tile': mgrs_tiles,
        'post_date': post_dates,
        'prod_date': prod_dates
    })

    return df_prod

def pull_hls_seq_local(start_date,end_date,hls_dir,bbox,mgrs_tile_id,prod_strs):
    # Get list of Dist-HLS in post time window

    minx, miny, maxx, maxy = bbox.bounds

    # Ensure output directory exists
    pathbase = Path(hls_dir)
    pathbase.mkdir(parents=True, exist_ok=True)

    collection_id = "OPERA_L3_DIST-ALERT-HLS_V1"

    # Date range in ISO 8601 format (UTC)
    #start_date = spd[0]
    #end_date = spd[-1]

    #start_date = df_prod['post_date'].min()
    #end_date = df_prod['post_date'].max()

    print("Look for correct Dist-HLS collection id")
    cmr_collections_url = "https://cmr.earthdata.nasa.gov/search/collections.json"
    params = {
        "keyword": "OPERA_L3_DIST-ALERT-HLS",  # or just "DIST-ALERT-HLS"
        "provider": "LPCLOUD",
        "page_size": 50,
    }
    r = requests.get(cmr_collections_url, params=params)
    r.raise_for_status()
    collections = r.json().get("feed", {}).get("entry", [])
    for col in collections:
        print(col["short_name"], col.get("version_id"), col["id"])

    collection_concept_id = "C2746980408-LPCLOUD"  # OPERA_L3_DIST-ALERT-HLS_V1
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "collection_concept_id": collection_concept_id,
        "temporal": f"{start_date},{end_date}",
        # match granule names like OPERA_L3_DIST-ALERT-HLS_T10TER_...
        "bounding_box": f"{minx},{miny},{maxx},{maxy}",
        "page_size": 200,
    }

    r = requests.get(cmr_url, params=params)
    r.raise_for_status()
    granules = r.json().get("feed", {}).get("entry", [])

    mgrs_prod_filtered = []
    for d in granules:
        title = d.get("title")
        if not title:
            continue
        parts = title.split("_")
        if len(parts) >= 4:
            if mgrs_tile_id in parts[3]:
                for link in d.get("links", []):
                    href = link.get("href","")
                    if not href.startswith("https://"):
                        continue
                    if any(sub in href for sub in prod_strs):
                        mgrs_prod_filtered.append(d)
        
    #mgrs_filtered = [
    #    d for d in granules
    #    if "title" in d
    #    and len(d["title"].split("_")) >= 3
    #    and mgrs_tile_id in d["title"].split("_")[3]
    #]

    hls_path = Path(hls_dir)
    hls_path.mkdir(parents=True, exist_ok=True)

    # Reuse connection
    session = requests.Session()
    for g in mgrs_prod_filtered:
        granule_id = g.get("title") or g.get("id")
        granule_path = hls_path / Path(granule_id)
        granule_path.mkdir(parents=True, exist_ok=True)
        for link in g.get("links", []):
            url = link["href"]
            if not any(sub in url for sub in prod_strs):
                continue

            filename = os.path.basename(url.split("?")[0])
            out_path = granule_path / filename
            if os.path.exists(out_path):
                continue  # already downloaded

            print(f"Downloading {filename}")
            with session.get(url, stream=True) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

def merge_post_dates(df_s1_prod,df_hls_prod):
    for df in (df_s1_prod,df_hls_prod):
        df["post_ts"] = pd.to_datetime(df["post_date"],
            format="%Y%m%dT%H%M%SZ", utc=True)

    # Keep original indexes in columns
    df_s1_prod = df_s1_prod.reset_index().rename(columns={"index": "idx1"})
    df_hls_prod = df_hls_prod.reset_index().rename(columns={"index": "idx2"})

    df_s1_prod = df_s1_prod.sort_values("post_ts")
    df_hls_prod = df_hls_prod.sort_values("post_ts")
    left_matches = pd.merge_asof(
        df_s1_prod,
        df_hls_prod[["idx2", "post_ts"]].rename(
            columns={"post_ts": "post_ts2"}),
        left_on="post_ts",
        right_on="post_ts2",
        direction="nearest"
    )
    right_matches = pd.merge_asof(
        df_hls_prod,
        df_s1_prod[["idx1", "post_ts"]].rename(
            columns={"post_ts": "post_ts1"}),
        left_on="post_ts",
        right_on="post_ts1",
        direction="nearest"
    )
    pairs_left = left_matches[["idx1", "idx2"]]
    pairs_right = right_matches[["idx1", "idx2"]]
    all_pairs = pd.concat([pairs_left, pairs_right],
        ignore_index=True).drop_duplicates()
    # sort chronologically using timestamps from df_s1_prod/df_hls_prod
    all_pairs = (all_pairs.assign(
        ts1=lambda x: df_s1_prod.set_index("idx1").loc[x["idx1"],
            "post_ts"].to_list(),
        ts2=lambda x: df_hls_prod.set_index("idx2").loc[x["idx2"],
            "post_ts"].to_list(),
        )
    )
    all_pairs["ts_min"] = all_pairs[["ts1", "ts2"]].min(axis=1)
    all_pairs = all_pairs.sort_values("ts_min").reset_index(
        drop=True)[["idx1", "idx2"]]

    return all_pairs

def subset_path_list(path_list,gdf1,width,hgt):
  new_list = []
  for i,rtc_path in enumerate(path_list):
    with rasterio.open(rtc_path) as rtc:
      # Form subset RTC around point of interest with width,hgt
      subset_profile,subset_rtc = subset_geotif(gdf1,width,hgt,rtc)
      if len(subset_rtc) != 0:
        # Write subset RTC geotif
        write_subset_geotif(rtc_path,subset_profile,subset_rtc)
        new_list.append(Path(str(rtc_path.parent / rtc_path.stem) +
            "_subset" + rtc_path.suffix))
        print(f"rtc_path = {rtc_path}")
        print(f"shape = {subset_rtc.shape}")
  return new_list

def subset_geotif(gdf1,width,hgt,gtif):
  gdf1_re = gdf1.to_crs(gtif.crs)
  x = gdf1_re.geometry.x[0]
  y = gdf1_re.geometry.y[0]
  row,col = gtif.index(x,y)
  rqst_window = Window(col-width/2,row-hgt/2,width,hgt)
  # Clip window to available data
  full_window = Window(0,0,gtif.width,gtif.height)
  if ((rqst_window.col_off >= gtif.width) or
      (rqst_window.col_off + rqst_window.width <= 0) or
      (rqst_window.row_off >= gtif.height) or
      (rqst_window.row_off + rqst_window.height <= 0)):
      window = Window(0,0,0,0)
      subset_profile = gtif.profile
      subset_data = gtif.read(1,window=window)
  else:
      window=rqst_window.intersection(full_window).round_offsets().round_lengths()
      subset_data = gtif.read(1,window=window)
      subset_profile = gtif.profile
      if np.isnan(subset_data).all():
        # No data in region of interest - set window to zero size
        window = Window(0,0,0,0)
        subset_data = gtif.read(1,window=window)
      else:
        # Form new rasterio object with updated metadata
        window_bounds = rasterio.windows.bounds(window, gtif.transform)
        new_transform = from_bounds(*window_bounds, window.width, window.height)
        subset_profile.update({
          'height': window.height,
          'width': window.width,
          'transform': new_transform
        })
        if width < 512 or hgt < 512:
          # Avoid inefficient tiling on small subsets
          subset_profile.pop('tiled', None)
          subset_profile.pop('blockxsize', None)
          subset_profile.pop('blockysize', None)

  #raise Exception("end of subset_geotif")
  return subset_profile,subset_data

def write_subset_geotif(in_path,subset_profile,subset_data):
  basename = in_path.stem
  basesuffix = in_path.suffix
  subset_name = basename + "_subset" + basesuffix
  subset_path = in_path.parent / subset_name
  #print(f'subset_path = {subset_path}')
  with rasterio.open(subset_path,'w', **subset_profile) as subset:
    subset.write(subset_data,1)
  #raise Exception("end of write_subset_geotif")

def write_subset_geotif_png(in_path,subset_profile,subset_data):
  basename = in_path.stem
  basesuffix = in_path.suffix
  subset_name = basename + "_subset" + basesuffix
  subset_path = in_path.parent / subset_name
  print(f'subset_path = {subset_path}')
  with rasterio.open(subset_path,'w', **subset_profile) as subset:
    subset.write(subset_data,1)
  png_name = basename + "_subset.png"
  png_path = in_path.parent / png_name
  arr_no_nan = np.nan_to_num(subset_data, nan=0.0)
  mindata = np.nanmin(arr_no_nan)
  maxdata = np.nanmax(arr_no_nan)
  arr_uint8 = (255 * (arr_no_nan - mindata) /
    (maxdata - mindata)).astype(np.uint8)
  im_uint8 = Image.fromarray(arr_uint8)
  im_uint8.save(png_path)

def run_dist_s1_workflow_subset(
    mgrs_tile_id: str,
    post_date: str | datetime,
    track_number: int,
    subset_bbox = None,
    post_date_buffer_days: int = DEFAULT_POST_DATE_BUFFER_DAYS,
    dst_dir: str | Path = DEFAULT_DST_DIR,
    input_data_dir: str | Path | None = DEFAULT_INPUT_DATA_DIR,
    memory_strategy: str = DEFAULT_MEMORY_STRATEGY,
    low_confidence_alert_threshold: float = DEFAULT_LOW_CONFIDENCE_ALERT_THRESHOLD,
    high_confidence_alert_threshold: float = DEFAULT_HIGH_CONFIDENCE_ALERT_THRESHOLD,
    src_water_mask_path: str | Path | None = DEFAULT_SRC_WATER_MASK_PATH,
    tqdm_enabled: bool = DEFAULT_TQDM_ENABLED,
    apply_water_mask: bool = DEFAULT_APPLY_WATER_MASK,
    lookback_strategy: str = DEFAULT_LOOKBACK_STRATEGY,
    max_pre_imgs_per_burst_mw: tuple[int, ...] | None = DEFAULT_MAX_PRE_IMGS_PER_BURST_MW,
    delta_lookback_days_mw: tuple[int, ...] | None = DEFAULT_DELTA_LOOKBACK_DAYS_MW,
    product_dst_dir: str | Path | None = DEFAULT_PRODUCT_DST_DIR,
    bucket: str | None = None,
    bucket_prefix: str = '',
    n_workers_for_despeckling: int = DEFAULT_N_WORKERS_FOR_DESPECKLING,
    n_workers_for_norm_param_estimation: int = DEFAULT_N_WORKERS_FOR_NORM_PARAM_ESTIMATION,
    device: str = DEFAULT_DEVICE,
    model_source: str = DEFAULT_MODEL_SOURCE,
    model_cfg_path: str | Path | None = DEFAULT_MODEL_CFG_PATH,
    model_wts_path: str | Path | None = DEFAULT_MODEL_WTS_PATH,
    stride_for_norm_param_estimation: int = DEFAULT_STRIDE_FOR_NORM_PARAM_ESTIMATION,
    batch_size_for_norm_param_estimation: int = DEFAULT_BATCH_SIZE_FOR_NORM_PARAM_ESTIMATION,
    model_compilation: bool = DEFAULT_MODEL_COMPILATION,
    interpolation_method: str = DEFAULT_INTERPOLATION_METHOD,
    apply_despeckling: bool = DEFAULT_APPLY_DESPECKLING,
    apply_logit_to_inputs: bool = DEFAULT_APPLY_LOGIT_TO_INPUTS,
    algo_config_path: str | Path | None = None,
    prior_dist_s1_product: str | Path | None = DEFAULT_PRIOR_DIST_S1_PRODUCT,
    model_dtype: str = DEFAULT_MODEL_DTYPE,
    use_date_encoding: bool = DEFAULT_USE_DATE_ENCODING,
    run_config_path: str | Path | None = None,
    n_anniversaries_for_mw: int = DEFAULT_N_ANNIVERSARIES_FOR_MW,
    no_day_limit: int = DEFAULT_NO_DAY_LIMIT,
    exclude_consecutive_no_dist: bool = DEFAULT_EXCLUDE_CONSECUTIVE_NO_DIST,
    percent_reset_thresh: int = DEFAULT_PERCENT_RESET_THRESH,
    no_count_reset_thresh: int = DEFAULT_NO_COUNT_RESET_THRESH,
    max_obs_num_year: int = DEFAULT_MAX_OBS_NUM_YEAR,
    confidence_upper_lim: int = DEFAULT_CONFIRMATION_CONFIDENCE_UPPER_LIM,
    confirmation_confidence_threshold: float = DEFAULT_CONFIRMATION_CONFIDENCE_THRESHOLD,
    metric_value_upper_lim: float = DEFAULT_METRIC_VALUE_UPPER_LIM,
    model_context_length: int | None = None,
) -> Path:

    run_config = run_dist_s1_sas_prep_workflow(
        mgrs_tile_id,
        post_date,
        track_number,
        post_date_buffer_days=post_date_buffer_days,
        dst_dir=dst_dir,
        input_data_dir=input_data_dir,
        memory_strategy=memory_strategy,
        low_confidence_alert_threshold=low_confidence_alert_threshold,
        high_confidence_alert_threshold=high_confidence_alert_threshold,
        tqdm_enabled=tqdm_enabled,
        apply_water_mask=apply_water_mask,
        lookback_strategy=lookback_strategy,
        max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
        delta_lookback_days_mw=delta_lookback_days_mw,
        src_water_mask_path=src_water_mask_path,
        product_dst_dir=product_dst_dir,
        bucket=bucket,
        bucket_prefix=bucket_prefix,
        n_workers_for_despeckling=n_workers_for_despeckling,
        n_workers_for_norm_param_estimation=n_workers_for_norm_param_estimation,
        device=device,
        model_source=model_source,
        model_cfg_path=model_cfg_path,
        model_wts_path=model_wts_path,
        stride_for_norm_param_estimation=stride_for_norm_param_estimation,
        batch_size_for_norm_param_estimation=batch_size_for_norm_param_estimation,
        model_compilation=model_compilation,
        interpolation_method=interpolation_method,
        apply_despeckling=apply_despeckling,
        apply_logit_to_inputs=apply_logit_to_inputs,
        algo_config_path=algo_config_path,
        prior_dist_s1_product=prior_dist_s1_product,
        model_dtype=model_dtype,
        use_date_encoding=use_date_encoding,
        run_config_path=run_config_path,
        n_anniversaries_for_mw=n_anniversaries_for_mw,
        no_day_limit=no_day_limit,
        exclude_consecutive_no_dist=exclude_consecutive_no_dist,
        percent_reset_thresh=percent_reset_thresh,
        no_count_reset_thresh=no_count_reset_thresh,
        max_obs_num_year=max_obs_num_year,
        confirmation_confidence_upper_lim=confidence_upper_lim,
        confirmation_confidence_threshold=confirmation_confidence_threshold,
        metric_value_upper_lim=metric_value_upper_lim,
        model_context_length=model_context_length,
    )

    df_bursts = get_burst_table_from_mgrs_tiles(run_config.mgrs_tile_id)

    #raise Exception("subset 1")

    # Identify bursts that intersect with the area of interest around a point
    pos_matches_index = df_bursts.sindex.query(subset_bbox)
    pos_matches = df_bursts.iloc[pos_matches_index]
    matches = pos_matches[pos_matches.intersects(subset_bbox)]

    # Use first copol RTC file to set projected CRS
    subset_gdf = gpd.GeoDataFrame({"geometry": [subset_bbox]},
        crs="EPSG:4326")
    with rasterio.open(run_config.post_rtc_copol[0]) as src:
        subset_gdf_px = bbox_pixels_from_gdf(subset_gdf,resolution_m = 30.0,
            projected_crs=src.crs)
    centroids = subset_gdf_px.geometry.centroid
    gdf1 = gpd.GeoDataFrame(
        subset_gdf_px.drop(columns="geometry"),  # keep non-geometry columns
        geometry=centroids,                      # new geometry = centroids
        crs=subset_gdf_px.crs                    # preserve CRS
    )
    width = subset_gdf_px['width_px'][0]
    hgt = subset_gdf_px['height_px'][0]

    print('Reducing to bursts covering area of interest')
    burst_ids = list(matches['jpl_burst_id'])
    #post_copol_filtered = [item for item in run_config.post_rtc_copol
    #  if any(burst_id in str(item) for burst_id in burst_ids)]
    #post_xpol_filtered = [item for item in run_config.post_rtc_crosspol
    #  if any(burst_id in str(item) for burst_id in burst_ids)]
    #pre_copol_filtered = [item for item in run_config.pre_rtc_copol
    #  if any(burst_id in str(item) for burst_id in burst_ids)]
    #pre_xpol_filtered = [item for item in run_config.pre_rtc_crosspol
    #  if any(burst_id in str(item) for burst_id in burst_ids)]

    # Filter out the local copol RTC paths in the subset
    # Copol and Crosspol are assumed to always correspond
    loc_path_copol_filtered = [
        item for item in run_config.df_inputs.loc_path_copol
            if any(burst_id in str(item) for burst_id in burst_ids)]
    if len(loc_path_copol_filtered) != 0:
        print('Downsizing to subarray of each burst')
        #run_config.post_rtc_copol = subset_path_list(post_copol_filtered,
        #    gdf1,width,hgt)
        #run_config.post_rtc_crosspol = subset_path_list(post_xpol_filtered,
        #    gdf1,width,hgt)
        #run_config.pre_rtc_copol = subset_path_list(pre_copol_filtered,
        #    gdf1,width,hgt)
        #run_config.pre_rtc_crosspol = subset_path_list(pre_xpol_filtered,
        #    gdf1,width,hgt)

        # Reduce the df_inputs dataframe to just the subset
        df_inputs = run_config.df_inputs
        new_df_inputs = df_inputs[
            df_inputs.loc_path_copol.isin(loc_path_copol_filtered)]
        # Assume crosspol corresopnds to copol
        loc_path_crosspol_filtered = new_df_inputs.loc_path_crosspol.tolist()

        # Build subset paths using modified mgrs_tile_id
        loc_path_copol_subset = replace_subfolder_component(
            loc_path_copol_filtered,mgrs_tile_id,mgrs_tile_id + '_subset')
        loc_path_crosspol_subset = replace_subfolder_component(
            loc_path_crosspol_filtered,mgrs_tile_id,mgrs_tile_id + '_subset')

        # Subset the filtered lists of RTC inputs and use the subset name list
        new_df_inputs.loc_path_copol = subset_list(
            loc_path_copol_filtered,loc_path_copol_subset,gdf1,width,hgt)
        new_df_inputs.loc_path_crosspol = subset_list(
            loc_path_crosspol_filtered,loc_path_crosspol_subset,gdf1,width,hgt)

        # Add extra columns needed to build a geodataframe object
        # which in turn is needed to construct a RunConfigData object
        new_df_inputs['url_copol'] = 'dummy string'
        new_df_inputs['url_crosspol'] = 'dummy string'
        new_df_inputs['geometry'] = subset_bbox
        gdf_product_loc = gpd.GeoDataFrame(new_df_inputs, geometry="geometry",
            crs="EPSG:4326")

        # Construct new RunConfigData object using just the subset
        new_run_config = RunConfigData.from_product_df(
            gdf_product_loc,
            dst_dir=dst_dir,
            max_pre_imgs_per_burst_mw=max_pre_imgs_per_burst_mw,
            model_context_length=model_context_length,
            delta_lookback_days_mw=delta_lookback_days_mw,
            lookback_strategy=lookback_strategy,
            model_source=model_source,
            model_cfg_path=model_cfg_path,
        )

        # Transfer extra columns not included in the above constructor
        new_run_config.product_dst_dir = run_config.product_dst_dir
        new_run_config.prior_dist_s1_product = run_config.prior_dist_s1_product

        raise Exception("subset stop 1")

        # Execute workflow on just the subset
        new_run_config2 = run_dist_s1_sas_workflow(new_run_config)

        # Note that subsetting the outputs has to wait until the
        # confirmation process is finished using the full size files
        # Alternatively could separate confirmation out of the
        # sas_workflow, but this seems easier for now.
        print('Subsetting output product files in prior product')
        if new_run_config2.prior_dist_s1_product is not None:
            p = new_run_config2.prior_dist_s1_product.dst_dir / (
                new_run_config2.prior_dist_s1_product.product_name)
            subset_outputs(p,gdf1,width,hgt)

        return new_run_config2,gdf1,width,hgt
    else:
        return run_config,gdf1,width,hgt

def bbox_pixels_from_gdf(gdf: gpd.GeoDataFrame,
    resolution_m,
    projected_crs="EPSG:3857"):
    """
    Compute width/height in pixels for each feature's bounding box,
    and add the center point geometry as a new column.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame CRS is not set; cannot interpret units.")

    # Reproject to a CRS in meters
    gdf_m = gdf.to_crs(projected_crs)  # units = meters

    if isinstance(resolution_m, (int, float)):
        res_x = res_y = float(resolution_m)
    else:
        res_x, res_y = map(float, resolution_m)

    width_px_list = []
    height_px_list = []
    center_points_m = []  # centroids in meter CRS

    for geom in gdf_m.geometry:
        minx, miny, maxx, maxy = geom.bounds
        width_m = maxx - minx
        height_m = maxy - miny

        width_px = int(round(width_m / res_x))
        height_px = int(round(height_m / res_y))

        width_px_list.append(width_px)
        height_px_list.append(height_px)

        # center point from bounds (equivalent to centroid for rectangles)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        from shapely.geometry import Point
        center_points_m.append(Point(cx, cy))

    # Build output in projected CRS
    gdf_m_out = gdf_m.copy()
    gdf_m_out["width_px"] = width_px_list
    gdf_m_out["height_px"] = height_px_list
    gdf_m_out["center_m"] = center_points_m  # point in projected CRS

    # reproject back to original CRS, including center points
    gdf_out = gdf_m_out.to_crs(gdf.crs)
    gdf_out = gdf_out.rename(columns={"center_m": "center"})
    return gdf_out

def id_bbox_setup(df_id,prop):
    min_lon = df_id["Long"].min()
    max_lon = df_id["Long"].max()
    min_lat = df_id["Lat"].min()
    max_lat = df_id["Lat"].max()
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    # Expand bounds by prop on each side
    min_lon_exp = min_lon - prop * lon_range
    max_lon_exp = max_lon + prop * lon_range
    min_lat_exp = min_lat - prop * lat_range
    max_lat_exp = max_lat + prop * lat_range
    bbox = box(min_lon_exp, min_lat_exp, max_lon_exp, max_lat_exp)
    gdf_bbox = gpd.GeoDataFrame({"geometry": [bbox]},
        crs="EPSG:4326")
    return bbox,gdf_bbox

def replace_subfolder_component(
    paths: Iterable[Path], old: str, new: str
) -> List[Path]:
    """
    Return new Paths where any directory component equal to `old`
    is replaced by `new`. Filenames are not modified.

    Args:
        paths: Iterable of Path objects.
        old: Folder name to search for (exact match of a path component).
        new: Replacement folder name.

    Returns:
        List of updated Path objects.
    """
    updated = []
    for p in paths:
        parts = p.parts

        # Separate directory components and filename
        dir_parts = parts[:-1]
        file_part = parts[-1] if parts else None

        # Replace only in directory components
        new_dir_parts = tuple(new
            if part == old else part for part in dir_parts)

        # Rebuild path
        if file_part is not None:
            new_path = Path(*new_dir_parts, file_part)
        else:
            new_path = Path(*new_dir_parts)
        updated.append(new_path)

    return updated
