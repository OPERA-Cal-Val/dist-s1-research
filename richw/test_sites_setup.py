#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1/bin/python -i

import os
import geopandas as gpd
from rasterio.crs import CRS
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
import data_fcns

filename_burst_id = 'opera_burst_ids.geojson.zip'
url_burst_id = 'https://github.com/opera-adt/burst_db/releases/download/v0.3.1/burst-id-geometries-simple-0.3.1.geojson.zip'
filename_slcs_for_processing = 'slcs_for_processing.csv.zip'
#url_maplabels = 'https://github.com/OPERA-Cal-Val/DIST-Validation/blob/main/mapLabelsv1sample'
url_maplabels_base = 'https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/mapLabelsv1sample'
filename_rtc_s1_table = 'rtc_s1_table.json.zip'

# Check for burst_id database and pull if needed
if os.path.isfile(filename_burst_id):
  print(f"Using burst_id database file: {filename_burst_id}")
else:
  print(f"Loading burst_id database from {url_burst_id}")
  df = gpd.read_file(url_burst_id)
  print(f"Writing burst_id database file to: {filename_burst_id}")
  to_geojson_gzip(df, filename_burst_id)

# Read optical reference data (human observer results)
# for the 300 test sites
df_dist_hls_val = pd.read_csv('https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/referenceTimeSeries_last.csv')

# Testing
IND_T = 2
row_data = list(df_dist_hls_val.iloc[IND_T])
ts_labeled = row_data[5:]
ind_c = data_fcns.get_first_change(ts_labeled)
ind_nc = data_fcns.get_last_noChange(ts_labeled, ind_c)

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

# Plot the test site locations
fig,ax = plt.subplots()
df_sites.plot(ax=ax)
fig.savefig('sites.png',dpi=300,bbox_inches="tight")

# Write the site table to a geojson file
df_sites.to_file('dist_hls_val_sites.geojson', driver='GeoJSON')

# Join burst table with site id table
# Following marshak/3_dist_sites/dist_hls_validation_table.ipynb

# Subset of sites
# Read the table made by CM
# marshak/3_dist_sites/dist_hls_validation_table.ipynb
df_slcs = pd.read_csv(filename_slcs_for_processing)
sites_used = df_slcs.site_id.unique()
df_sites_subset = df_sites[df_sites.site_id.isin(sites_used)].reset_index(drop=True)
# Write site subset table
df_sites_subset.to_file('sites_for_processing_may_2024.geojson', driver='GeoJSON')

# Load RTC Table
# Read the table made by CM
# marshak/4_rtc_organization/0_Organize-RTC-Data.ipynb
df_rtc = pd.read_json(filename_rtc_s1_table)

# Map labels file - contains optical algorithm outputs for the 300 test sites
filename_maplabels1 = '1_DIST-ALERT_v1sample.csv'
filename_maplabels4 = '4_DIST-ALERT_v1sample.csv'
maplabel1 = pd.read_csv(url_maplabels_base + '/' + filename_maplabels1)
maplabel4 = pd.read_csv(url_maplabels_base + '/' + filename_maplabels4)

print('Iterating over site subset')
veg_dist_date = []
gen_dist_date = []
ref_dist_date = []
fig1,ax1 = plt.subplots()

for index, row in df_sites_subset.iterrows():
#index, row = next(df_sites_subset.iterrows())
  print(row['site_id'])

  ref_dist_date1 = row['change_time']
  ref_dist_date.append(ref_dist_date1)

  site_id_str = "{:.0f}".format(row['site_id'])
  filename_maplabel = site_id_str + '_DIST-ALERT_v1sample.csv'
  maplabel = pd.read_csv(url_maplabels_base + '/' + filename_maplabel)
  dates = maplabel['VEG-DIST-DATE']
  date_present = dates.notna()
  i_found_date = list(itertools.compress(range(len(date_present)),date_present))
  found_date = list(itertools.compress(dates,date_present))
  if len(found_date) == 0:
    veg_dist_date.append(0)
  else:
    first_date = found_date[0]
    i_first = i_found_date[0]
    veg_dist_date.append(first_date)

  dates = maplabel['GEN-DIST-DATE']
  date_present = dates.notna()
  i_found_date = list(itertools.compress(range(len(date_present)),date_present))
  found_date = list(itertools.compress(dates,date_present))
  if len(found_date) == 0:
    gen_dist_date.append(0)
  else:
    first_date = found_date[0]
    i_first = i_found_date[0]
    gen_dist_date.append(first_date)

