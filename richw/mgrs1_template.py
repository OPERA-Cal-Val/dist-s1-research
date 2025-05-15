#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1-env/bin/python -i

import sys
import os
import inspect
from pathlib import Path

from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from PIL import Image
import numpy as np

import plot_fcns

from dist_s1.workflows import (
    run_despeckle_workflow,
    run_normal_param_estimation_workflow,
    run_burst_disturbance_workflow,
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
  run_config_template_path = 'rc1.yml'
  run_config = run_dist_s1_sas_prep_runconfig_yml(run_config_template_path)

  # Point of interest (WGS84 reference)
  lat1 = 34.17441349209357
  lon1 = -118.09816512175193
  width = 50
  hgt = 50
  point = Point(lon1,lat1)
  # Identify bursts from this MGRS tile that cover point of interest
  df_bursts = get_burst_table_from_mgrs_tiles(run_config.mgrs_tile_id) 
  possible_matches_index = df_bursts.sindex.query(point)
  possible_matches = df_bursts.iloc[possible_matches_index]
  matches = possible_matches[possible_matches.contains(point)]
  gdf1 = gpd.GeoDataFrame({'Name': ['Point of Interest'], 'geometry': [point]},
    crs="EPSG:4326")

  print('Reducing to one burst')
  burst_id = matches.iloc[0]['jpl_burst_id']
  post_copol_filtered = [item for item in run_config.post_rtc_copol
    if burst_id in str(item)]
  post_xpol_filtered = [item for item in run_config.post_rtc_crosspol
    if burst_id in str(item)]
  pre_copol_filtered = [item for item in run_config.pre_rtc_copol
    if burst_id in str(item)]
  pre_xpol_filtered = [item for item in run_config.pre_rtc_crosspol
    if burst_id in str(item)]
 
  all_inputs = post_copol_filtered + post_xpol_filtered + pre_copol_filtered + pre_xpol_filtered

  print('Downsizing to subarray of one burst')
  for i,rtc_path in enumerate(all_inputs):
    with rasterio.open(rtc_path) as rtc:
      # Form subset RTC around point of interest with width,hgt
      subset_profile,subset_rtc = subset_geotif(gdf1,width,hgt,rtc)
      # Write subset RTC geotif
      write_subset_geotif(rtc_path,subset_profile,subset_rtc)

  # Use the single burst subset in the runconfig lists
  run_config.post_rtc_copol = [Path(str(p.parent / p.stem) + "_subset"
    + p.suffix) for p in post_copol_filtered]
  run_config.post_rtc_crosspol = [Path(str(p.parent / p.stem) + "_subset"
    +  p.suffix) for p in post_xpol_filtered]
  run_config.pre_rtc_copol = [Path(str(p.parent / p.stem) + "_subset"
    + p.suffix) for p in pre_copol_filtered]
  run_config.pre_rtc_crosspol = [Path(str(p.parent / p.stem) + "_subset"
    + p.suffix) for p in pre_xpol_filtered]

  #print('despeckling')
  #run_despeckle_workflow(run_config)
  print('sas processing')
  new_run_config = run_dist_s1_processing_workflow(run_config)

  # Subset the output product files
  print('Downsizing output products into .png files')
  for i,out_path in enumerate(run_config.final_unformatted_tif_paths.values()):
    with rasterio.open(out_path) as out:
      # Form subset output product around point of interest with width,hgt
      subset_profile,subset_out = subset_geotif(gdf1,width,hgt,out)
      # Write subset output png
      write_subset_geotif_png(out_path,subset_profile,subset_out)

  raise Exception("End of main")

def subset_geotif(gdf1,width,hgt,gtif):
  gdf1_re = gdf1.to_crs(gtif.crs)
  x = gdf1_re.geometry.x[0]
  y = gdf1_re.geometry.y[0]
  row,col = gtif.index(x,y)
  rqst_window = Window(col-width/2,row-hgt/2,width,hgt)
  # Clip window to available data
  full_window = Window(0,0,gtif.width,gtif.height)
  window = rqst_window.intersection(full_window)
  subset_data = gtif.read(1,window=window)
  # Form new raserio object with updated metadata
  window_bounds = rasterio.windows.bounds(window, gtif.transform)
  new_transform = from_bounds(*window_bounds, window.width, window.height)
  subset_profile = gtif.profile
  subset_profile.update({
    'height': window.height,
    'width': window.width,
    'transform': new_transform
  })
  return subset_profile,subset_data

def write_subset_geotif(in_path,subset_profile,subset_data):
  basename = in_path.stem
  basesuffix = in_path.suffix
  subset_name = basename + "_subset" + basesuffix
  subset_path = in_path.parent / subset_name
  print(f'subset_path = {subset_path}')
  with rasterio.open(subset_path,'w', **subset_profile) as subset:
    subset.write(subset_data,1)

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
  arr_uint8 = (255 * (subset_data - subset_data.min()) /
    (np.ptp(subset_data) + 1e-8)).astype(np.uint8)
  im_uint8 = Image.fromarray(arr_uint8)
  im_uint8.save(png_path)

if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    exc_type,exc_value,exc_traceback = sys.exc_info()
    tb = exc_traceback
    stack = inspect.trace()
    lcls = locals()
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

