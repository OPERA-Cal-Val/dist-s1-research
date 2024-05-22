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
import data_fcns

filename_burst_id = 'opera_burst_ids.geojson.zip'
url_burst_id = 'https://github.com/opera-adt/burst_db/releases/download/v0.3.1/burst-id-geometries-simple-0.3.1.geojson.zip'
filename_slcs_for_processing = 'slcs_for_processing.csv.zip'

# Check for burst_id database and pull if needed
if os.path.isfile(filename_burst_id):
  print(f"Using burst_id database file: {filename_burst_id}")
else:
  print(f"Loading burst_id database from {url_burst_id}")
  df = gpd.read_file(url_burst_id)
  print(f"Writing burst_id database file to: {filename_burst_id}")
  to_geojson_gzip(df, filename_burst_id)

# Read disturbance data
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

df_sites = gpd.GeoDataFrame({'site_id': df_dist_hls_val.ID,
  'change_label': df_dist_hls_val.overallLabel,
  'change_time': change_times,
  'last_observation_time': last_obs_times},
  geometry=gpd.points_from_xy(df_dist_hls_val.Long,
  df_dist_hls_val.Lat),
  crs=CRS.from_epsg(4326))

# Plot the test site locations
fig,ax = plt.subplots()
df_sites.plot(ax=ax)
fig.savefig('sites.png',dpi=300,bbox_inches="tight")

df_sites.to_file('dist_hls_val_sites.geojson', driver='GeoJSON')

# Subset of sites
df_slcs = pd.read_csv(filename_slcs_for_processing)
sites_used = df_slcs.site_id.unique()
df_sites_subset = df_sites[df_sites.site_id.isin(sites_used)].reset_index(drop=True)
df_sites_subset.to_file('sites_for_processing_may_2024.geojson', driver='GeoJSON')
