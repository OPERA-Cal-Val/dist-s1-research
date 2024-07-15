from pptx import Presentation
from pptx.util import Inches
import os
import sys
import inspect
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
from dem_stitcher.geojson_io import read_geojson_gzip
from dem_stitcher.geojson_io import to_geojson_gzip
from shapely.geometry import Polygon, shape
import asf_search as asf
import concurrent.futures
from tqdm import tqdm
import itertools
from datetime import datetime, timedelta
from dateutil.parser import parse
import ast
#import distmetrics
import data_fcns
import plot_fcns

def setup(dir_base):

  dir_dist_s1_research = dir_base + '/dist-s1-research'
  dir_dist_s1_validation_harness = dir_base + '/dist-s1-validation-harness'
  dir_rtc_tables = dir_dist_s1_validation_harness + '/data'

  filename_burst_id = 'opera_burst_ids.geojson.zip'
  url_burst_id = 'https://github.com/opera-adt/burst_db/releases/download/v0.3.1/burst-id-geometries-simple-0.3.1.geojson.zip'
  filename_slcs_for_processing = 'slcs_for_processing.csv.zip'
  #url_maplabels = 'https://github.com/OPERA-Cal-Val/DIST-Validation/blob/main/mapLabelsv1sample'
  url_maplabels_base = 'https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/mapLabelsv1sample'
  filename_rtc_s1_table = 'rtc_s1_table.json.zip'
  dir_rtc_data = '/u/aurora-r0/cmarshak/dist-s1-research/marshak/6_torch_dataset/opera_rtc_data'
  #dir_rtc_site_data = dir_dist_s1_research + '/oliver/rtc_analysis/tables'
  dir_rtc_site_data = dir_dist_s1_validation_harness + '/data/rtc_ts_by_site'
  print(f"Using rtc site data from: {dir_rtc_site_data}")
  basename_rtc_site_data = 'rtc_summary_burst_'

  url_maplabels_base = 'https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/mapLabelsv1sample'
  ALERTname = "v1sample"

  # Check for burst_id database and pull if needed
  if os.path.isfile(filename_burst_id):
    print(f"Using burst_id database file: {filename_burst_id}")
    df_burst = read_geojson_gzip(filename_burst_id)
  else:
    print(f"Loading burst_id database from {url_burst_id}")
    df_burst = gpd.read_file(url_burst_id)
    print(f"Writing burst_id database file to: {filename_burst_id}")
    to_geojson_gzip(df_burst, filename_burst_id)

  # Read optical reference data (human observer results)
  # for the 300 test sites
  df_dist_hls_val = pd.read_csv('https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/referenceTimeSeries_last.csv')

  change_times = df_dist_hls_val.apply(data_fcns.get_first_change_from_row,
  axis=1)
  last_obs_times = df_dist_hls_val.apply(
  data_fcns.get_last_obs_date_before_change, axis=1)
  change_labels = df_dist_hls_val.overallLabel[:10]

  # Construct site table for the 300 test sites
  # Include all needed metadata
  df_sites = gpd.GeoDataFrame({'site_id': df_dist_hls_val.ID,
    'change_label': df_dist_hls_val.overallLabel,
    'change_type': df_dist_hls_val.changetype,
    'change_time': change_times,
    'last_observation_time': last_obs_times},
    geometry=gpd.points_from_xy(df_dist_hls_val.Long,
    df_dist_hls_val.Lat),
    crs=CRS.from_epsg(4326))
  
  # Write the site table to a geojson file
  df_sites.to_file('dist_hls_val_sites.geojson', driver='GeoJSON')

  # Join burst table with site id table
  # Following marshak/3_dist_sites/dist_hls_validation_table.ipynb
  df_val_bursts = gpd.sjoin(df_burst, df_sites, how='inner',
    predicate='intersects').reset_index(drop=True)
  df_val_bursts = df_val_bursts.drop_duplicates()
  df_val_bursts = df_val_bursts.drop(columns=['index_right'])
  df_val_bursts['track_number'] = df_val_bursts.burst_id_jpl.map(
    lambda burst_id_jpl: int(burst_id_jpl.split('_')[0][1:]))
  df_val_bursts = df_val_bursts.sort_values(by=['site_id',
    'burst_id_jpl']).reset_index(drop=True)
  df_val_bursts.rename(columns={'burst_id_jpl': 'jpl_burst_id'}, inplace=True)

  #df_val_bursts['jpl_burst_id'] = df_val_bursts.applymap(lambda x: 
  for index,row in df_val_bursts.iterrows():
    # Convert jpl_burst_id (uppercase,dashes) 
    burst_id = row['jpl_burst_id']
    burst_id_d = burst_id.replace("_","-")
    burst_id_upper = burst_id_d.upper()
    #df_val_bursts.at['jpl_burst_id',index] = burst_id_upper
    df_val_bursts['jpl_burst_id'][index] = burst_id_upper
    #df_val_bursts['jpl_burst_id'].replace(burst_id,burst_id_upper)
  
  # Subset of sites
  # Read the table made by CM
  # marshak/3_dist_sites/dist_hls_validation_table.ipynb
  df_slcs = pd.read_csv(filename_slcs_for_processing)
  sites_used = df_slcs.site_id.unique()
  df_sites_subset = df_sites[df_sites.site_id.isin(sites_used)].reset_index(drop=True)
  # Write site subset table
  df_sites_subset.to_file('sites_for_processing_may_2024.geojson', driver='GeoJSON')
  df_val_bursts_subset = df_val_bursts[
    df_val_bursts.site_id.isin(sites_used)].reset_index(drop=True)
  df_val_bursts_subset.to_file('val_bursts.geojson', driver='GeoJSON')

  # Load RTC Table
  # Read the table made by CM
  # marshak/4_rtc_organization/0_Organize-RTC-Data.ipynb
  df_rtc = pd.read_json(filename_rtc_s1_table)

  # Note that df_val_bursts_subset has multiple bursts for one site,
  # however, only one will be present in df_rtc.  When selecting the bursts,
  # only one per site was chosen (the one with the site near the middle).

  return df_rtc,df_val_bursts_subset,df_sites_subset,dir_rtc_site_data,basename_rtc_site_data

