#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1-env/bin/python -i

import sys
import os
import inspect
from pathlib import Path

import math
from shapely.geometry import Point
from shapely.geometry import box
from geopy.distance import geodesic
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from PIL import Image
import numpy as np
import warnings

import asf_search

import plot_fcns

from dist_s1.workflows import (
    run_despeckle_workflow,
    run_normal_param_estimation_workflow,
    run_burst_disturbance_workflow,
    run_disturbance_confirmation,
    run_dist_s1_localization_workflow,
    run_dist_s1_workflow,
    run_dist_s1_processing_workflow,
    run_dist_s1_sas_prep_workflow,
    run_dist_s1_sas_prep_runconfig_yml,
    run_dist_s1_sas_workflow,
    run_disturbance_merge_workflow,
    package_disturbance_tifs
)

from dist_s1_enumerator import(
  get_burst_table_from_mgrs_tiles
)

# Need to use main() when using multiprocessing!
def main() -> None:

  print('sas prep')
  asf_search.constants.INTERNAL.CMR_TIMEOUT = 90

  run_config_template_path = 'rc1.yml'
  run_config = run_dist_s1_sas_prep_runconfig_yml(run_config_template_path)

  # Point of interest (WGS84 reference)
  lat1 = 34.17441349209357
  lon1 = -118.09816512175193
  width = 600
  hgt = 400
  
  width_meters = width*30
  hgt_meters = hgt*30
  diag_meters = math.sqrt(width_meters*width_meters + hgt_meters*hgt_meters)
  point = Point(lon1,lat1)
  bearing_ne = math.atan2(hgt_meters,width_meters)
  bearing_ne_deg = math.degrees(bearing_ne)
  gd1 = geodesic(meters=diag_meters)
  corner_ne = gd1.destination((lat1,lon1),bearing_ne_deg)
  bearing_sw = math.atan2(-hgt_meters,-width_meters)
  bearing_sw_deg = math.degrees(bearing_sw)
  corner_sw = gd1.destination((lat1,lon1),bearing_sw_deg)
  bbox = box(corner_sw.longitude,corner_sw.latitude,
      corner_ne.longitude,corner_ne.latitude)
  df_bursts = get_burst_table_from_mgrs_tiles(run_config.mgrs_tile_id) 

  # Identify bursts that intersect with the area of interest around a point
  pos_matches_index = df_bursts.sindex.query(bbox)
  pos_matches = df_bursts.iloc[pos_matches_index]
  matches = pos_matches[pos_matches.intersects(bbox)]

  # Identify bursts from this MGRS tile that cover point of interest
  possible_matches_index = df_bursts.sindex.query(point)
  possible_matches = df_bursts.iloc[possible_matches_index]
  matches1 = possible_matches[possible_matches.contains(point)]

  gdf1 = gpd.GeoDataFrame({'Name': ['Point of Interest'], 'geometry': [point]},
    crs="EPSG:4326")

  #print('Reducing to one burst')
  #burst_id = matches.iloc[0]['jpl_burst_id']
  #post_copol_filtered = [item for item in run_config.post_rtc_copol
  #  if burst_id in str(item)]
  #post_xpol_filtered = [item for item in run_config.post_rtc_crosspol
  #  if burst_id in str(item)]
  #pre_copol_filtered = [item for item in run_config.pre_rtc_copol
  #  if burst_id in str(item)]
  #pre_xpol_filtered = [item for item in run_config.pre_rtc_crosspol
  #  if burst_id in str(item)]

  print('Reducing to bursts covering area of interest')
  burst_ids = list(matches['jpl_burst_id'])
  post_copol_filtered = [item for item in run_config.post_rtc_copol
    if any(burst_id in str(item) for burst_id in burst_ids)]
  post_xpol_filtered = [item for item in run_config.post_rtc_crosspol
    if any(burst_id in str(item) for burst_id in burst_ids)]
  pre_copol_filtered = [item for item in run_config.pre_rtc_copol
    if any(burst_id in str(item) for burst_id in burst_ids)]
  pre_xpol_filtered = [item for item in run_config.pre_rtc_crosspol
    if any(burst_id in str(item) for burst_id in burst_ids)]
 
  all_inputs = post_copol_filtered + post_xpol_filtered + pre_copol_filtered + pre_xpol_filtered

  print('Downsizing to subarray of each burst')
  run_config.post_rtc_copol = subset_path_list(post_copol_filtered,
      gdf1,width,hgt)
  run_config.post_rtc_crosspol = subset_path_list(post_xpol_filtered,
      gdf1,width,hgt)
  run_config.pre_rtc_copol = subset_path_list(pre_copol_filtered,
      gdf1,width,hgt)
  run_config.pre_rtc_crosspol = subset_path_list(pre_xpol_filtered,
      gdf1,width,hgt)

  #for i,rtc_path in enumerate(all_inputs):
  #  with rasterio.open(rtc_path) as rtc:
  #    # Form subset RTC around point of interest with width,hgt
  #    subset_profile,subset_rtc = subset_geotif(gdf1,width,hgt,rtc)
  #    # Write subset RTC geotif
  #    write_subset_geotif(rtc_path,subset_profile,subset_rtc)

  # Use the subsetted burst subset in the runconfig lists
  #run_config.post_rtc_copol = [Path(str(p.parent / p.stem) + "_subset"
  #  + p.suffix) for p in post_copol_filtered]
  #run_config.post_rtc_crosspol = [Path(str(p.parent / p.stem) + "_subset"
  #  +  p.suffix) for p in post_xpol_filtered]
  #run_config.pre_rtc_copol = [Path(str(p.parent / p.stem) + "_subset"
  #  + p.suffix) for p in pre_copol_filtered]
  #run_config.pre_rtc_crosspol = [Path(str(p.parent / p.stem) + "_subset"
  #  + p.suffix) for p in pre_xpol_filtered]

  #print('despeckling')
  #run_despeckle_workflow(run_config)
  #with warnings.catch_warnings():
    #warnings.simplefilter("error", category=RuntimeWarning)
  print('sas processing')
  new_run_config = run_dist_s1_processing_workflow(run_config)
  if run_config.confirmation_strategy == 'use_prev_product':
    print('Using previous product for confirmation')
    run_disturbance_confirmation(new_run_config)

  # Subset the output product files
  print('Downsizing output products into .png files')
  for i,out_path in enumerate(new_run_config.final_unformatted_tif_paths.values()):
    with rasterio.open(out_path) as out:
      # Form subset output product around point of interest with width,hgt
      subset_profile,subset_out = subset_geotif(gdf1,width,hgt,out)
      # Write subset output png
      write_subset_geotif_png(out_path,subset_profile,subset_out)

  raise Exception("End of main")

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
        # Form new raserio object with updated metadata
        window_bounds = rasterio.windows.bounds(window, gtif.transform)
        new_transform = from_bounds(*window_bounds, window.width, window.height)
        subset_profile.update({
          'height': window.height,
          'width': window.width,
          'transform': new_transform
        })
        if width < 512 or hgt < 512:
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

class DoublyNode:
  def __init__(self,tb):
    self.tb = tb
    self.next = None
    self.prev = None

def up(dtb):
  if dtb.prev:
    dtb_new = dtb.prev
  else:
    dtb_new = dtb

  sl_new = dtb_new.tb.tb_frame.f_locals
  return dtb_new,sl_new

def down(dtb):
  if dtb.next:
    dtb_new = dtb.next
  else:
    dtb_new = dtb

  sl_new = dtb_new.tb.tb_frame.f_locals
  return dtb_new,sl_new

if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    exc_type,exc_value,exc_traceback = sys.exc_info()
    tb = exc_traceback
    stack = inspect.trace()
    lcls = locals()
    dtb = DoublyNode(tb)
    prev = dtb
    tb_current = tb.tb_next
    while tb_current:
      new_node = DoublyNode(tb_current)
      prev.next = new_node
      new_node.prev = prev
      prev = new_node
      tb_current = tb_current.tb_next

    dtb1 = dtb
    sl1 = dtb1.tb.tb_frame.f_locals
    while True:
      if dtb1.next:
        fname = dtb1.tb.tb_next.tb_frame.f_code.co_filename
        if "opera" in fname:
          dtb1,sl1 = down(dtb1)
        else:
          break
      else:
        break
    
    tb1 = tb
    while True:
      # Sweep for bottom of traceback list within user code
      # (where the error occurred)
      if tb1.tb_next:
        fname = tb1.tb_next.tb_frame.f_code.co_filename
        if "opera" in fname:
          tb1 = tb1.tb_next
        else:
          break
      else:
        break
    sl = tb1.tb_frame.f_locals
    fname = tb1.tb_frame.f_code.co_filename
    lineno = tb1.tb_frame.f_lineno
    #print(f"Error at mgrs_tile_id: {mgrs_tile_id}, track_number: {track_number}")
    print(f"Error at user line: {lineno} in {fname}")
    print(exc_value)
    rc = sl['run_config']
    #traceback.print_tb(tb)

#print('localization')
#run_config = run_dist_s1_localization_workflow(
#    mgrs_tile_id,
#    post_date,
#    track_number,
#    1,
#    dst_dir=dst_dir,
#    input_data_dir=dst_dir,
#)

#print('despeckling')
#run_config.memory_strategy = memory_strategy
#run_config.device = 'cpu'
#run_config.batch_size_for_despeckling = 50
#run_config.n_workers_for_despeckling = 1
#run_despeckle_workflow(run_config)
#print('normal param estimation')
#run_normal_param_estimation_workflow(runconfig)
#print('burst disturbance')
#run_burst_disturbance_workflow(run_config)
#print('disturbance merge')
#run_disturbance_merge_workflow(run_config)
#print('package tifs')
#package_disturbance_tifs(run_config)
#print('product data checks')
#product_data = run_config.product_data_model
#product_data.validate_tif_layer_dtypes()
#product_data.validate_layer_paths()

