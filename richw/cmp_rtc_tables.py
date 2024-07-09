#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1/bin/python -i

# Compare RTC data pulled from downloaded RTC burst data with
# tabulated data from dist-s1-research/oliver/rtc_analysis/tables

import os
import sys
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
import umd_fcns

dir_base = '/home/richw/src/opera'
dir_dist_s1_research = dir_base + '/dist-s1-research'
dir_dist_s1_validation_harness = dir_base + '/dist-s1-validation-harness'

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

print('Iterating over site subset')

fstr = "%Y%m%dT%H%M%SZ"
fstr2 = "%Y-%m-%d %H:%M:%S"

delta_dt_all = []
delta_dt_all_maxabs = []
delta_vv_all = []
delta_vv_all_maxabs = []
delta_vh_all = []
delta_vh_all_maxabs = []
max_delta_dt_allsites = []
max_delta_vv_allsites = []
max_delta_vh_allsites = []
nan_count = []
lendat_mismatch_allsites = []
site_ids_compared = []
burst_ids_compared = []
delta_10min = datetime(2022,11,26,14,10,0) - datetime(2022,11,26,14,0,0)


for index, df_row in df_val_bursts_subset.iterrows():
#for ii in range(0,47):
#  index,df_row = next(df_val_bursts_subset.iterrows())
  # Need burst id using uppercase and dashes
  burst_id = df_row['jpl_burst_id']
  site_id = df_row['site_id']
  # if this csv file does not exist, it's data file will also have not data
  filename_rtc_site_data = (dir_rtc_site_data + '/' + basename_rtc_site_data +
    burst_id + '_site' + str(site_id) + '.csv')
  df_site = df_sites_subset[df_sites_subset.site_id
    == site_id].reset_index(drop=True)
  lon = df_site.geometry.x[0]
  lat = df_site.geometry.y[0]
  # Site locations use long,lat defined on WGS84 ellipsoid which is the basis
  # of the EPSG:4326 coordinate reference system (CRS)
  gdf1 = gpd.GeoDataFrame(df_site.geometry,crs="EPSG:4326")
  # RTC data stored in directories named by burst id (uppercase,dashes)
  dir_burst1 = os.path.join(dir_rtc_data,burst_id)
  # RTC dataframe also uses burst id label (uppercase,dashes)
  # Note: order of entries of df_rtc_ts differs from order of filename_rtc
  df_rtc_ts = df_rtc[df_rtc.jpl_burst_id == burst_id].reset_index(drop=True)
  if len(df_rtc_ts) == 0:
    # This jpl_burst_id does not have data for this site_id
    print(f"id: {df_row['site_id']}, jpl_burst_id: {df_row['jpl_burst_id']} not present")
    continue
  else:
    print(f"id: {df_row['site_id']}, jpl_burst_id: {df_row['jpl_burst_id']}")
    if os.path.isfile(filename_rtc_site_data):
      df_rtc_site = pd.read_csv(filename_rtc_site_data)
    else:
      print("corresponding csv file not found - skipping comparison")
      continue

  # Reset lists that accumulate time series data for each burst
  vv_data = []
  vh_data = []
  vv_datetime = []
  vh_datetime = []
  vv_tbl_data = []
  vh_tbl_data = []
  a = df_rtc_site['datetime']
  adt = [datetime.strptime(x,fstr2) for x in a]
  tbl_dt = []
  # Each file has data from one datetime.  Iterate over all.
  for filepath_rtc_vv in Path(dir_burst1).glob("*" + burst_id + "*VV.tif"):
    # Enforce same datetime in VV and VH filenames
    filename_rtc_vv = filepath_rtc_vv.name
    filename_rtc_vh = filename_rtc_vv.replace('VV','VH')
    for filename_rtc in [filename_rtc_vv,filename_rtc_vh]:
      file_path = dir_burst1 + "/" + filename_rtc
      if not os.path.isfile(file_path):
        print(f"Error: {file_path} does not exist")
      with rasterio.open(file_path) as rtc_dataset:
        str_channel = filename_rtc[-6:-4]
        str_datetime = filename_rtc[32:48]
        datetime1 = datetime.strptime(str_datetime,fstr)
        # Find the closest match between the table datetime and the file
        # datetime.  This handles those filepath_rtc_vv values that are
        # out of chronological order (occasional pairs are flipped)
        i_near = min(range(len(a)), key=lambda i: abs(adt[i] - datetime1))
        crs = rtc_dataset.crs
        # Reproject site location into current CRS
        gdf1_re = gdf1.to_crs(crs)
        x = gdf1_re.geometry.x[0]
        y = gdf1_re.geometry.y[0]
        row,col = rtc_dataset.index(x,y)
        # The window will actually load a chunk which is 512x512, and then
        # subselect the 3x3 matrix centered on row,col
        window = Window(col-1,row-1,3,3)
        band3x3 = rtc_dataset.read(1,window=window)
        if str_channel == "VV":
          vv_data.append(band3x3)
          vv_datetime.append(datetime1)
          csv_str = df_rtc_site['vv'][i_near]
          csv_str = csv_str.replace('nan','None')
          vv_tbl_data.append(ast.literal_eval(csv_str))
          # Table only has one datetime, so store it only when VV data handled
          tbl_dt.append(parse(df_rtc_site['datetime'][i_near]))
          delta_dt1 = tbl_dt[-1] - datetime1
          if delta_dt1 > delta_10min:
            print(f"index = {index}")
            print(f"delta_dt1 = {delta_dt1}")
            print(f"file_path = {file_path}")
            print(f"datetime1 = {datetime1}")
            print(f"tbl_dt[-1] = {tbl_dt[-1]}")
            print(f"site_id = {site_id}")
            print(f"burst_id = {burst_id}")
            sys.exit()
        elif str_channel == "VH":
          vh_data.append(band3x3)
          vh_datetime.append(datetime1)
          csv_str = df_rtc_site['vh'][i_near]
          csv_str = csv_str.replace('nan','None')
          vh_tbl_data.append(ast.literal_eval(csv_str))

  # Iterate over datetimes again to compute differences between tables
  # and binary files
  found_nan = 0
  lendat_mismatch = 0
  for i,(dt_vv,dt_vh,dt_tbl) in enumerate(zip(vv_datetime,vh_datetime,tbl_dt)):
    if dt_vv != dt_vh:
      print(f"{i}: {dt_vv}, {dt_vh}")
    delta_dt = dt_vv - dt_tbl
    delta_dt_all.append(delta_dt)
    if np.isnan(np.sum(vv_data[i])):
      found_nan = found_nan + 1
    else:
      dat1 = vv_data[i].flatten()
      lendat1 = len(dat1)
      dat2 = vv_tbl_data[i]
      lendat2 = len(dat2)
      if lendat1 != lendat2:
        lendat_mismatch = lendat_mismatch + 1
      Ncmp = min(lendat1,lendat2)
      delta_vv = dat1[0:Ncmp] - dat2[0:Ncmp]
      delta_vv_all.append(delta_vv)
      delta_vv_all_maxabs.append(max([abs(ele) for ele in delta_vv]))
      dat1 = vh_data[i].flatten()
      lendat1 = len(dat1)
      dat2 = vh_tbl_data[i]
      lendat2 = len(dat2)
      Ncmp = min(lendat1,lendat2)
      delta_vh = dat1[0:Ncmp] - dat2[0:Ncmp]
      delta_vh_all.append(delta_vh)
      delta_vh_all_maxabs.append(max([abs(ele) for ele in delta_vh]))

  # Accumulate maximum deltas in datetime and in vv,vh backscatter
  delta_dt_all_maxabs.append(max([abs(ele) for ele in delta_dt_all]))
  max_delta_dt_allsites.append(max(delta_dt_all_maxabs))
  lendat_mismatch_allsites.append(lendat_mismatch)
  site_ids_compared.append(site_id)
  burst_ids_compared.append(burst_id)
  if found_nan == 0:
    nan_count.append(0)
    max_delta_vv_allsites.append(max(delta_vv_all_maxabs))
    max_delta_vh_allsites.append(max(delta_vh_all_maxabs))
  else:
    nan_count.append(found_nan)
    max_delta_vv_allsites.append(0.0)
    max_delta_vh_allsites.append(0.0)

