#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1/bin/python -i

from pptx import Presentation
from pptx.util import Inches
import pickle
import os
import sys
import inspect
import traceback
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
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
import yaml

import distmetrics
import data_fcns
import plot_fcns
import test_setup
import alg_fcns
from alg_fcns import get_rtc_fromfile
from distmetrics import compute_mahalonobis_dist_1d
from distmetrics import compute_mahalonobis_dist_2d
from plot_fcns import prs_implot2

dir_base = '/home/richw/src/opera'
dir_research = dir_base + '/dist-s1-research'
dir_ad_hoc = dir_research + '/marshak/10_ad_hoc_data_generation'
dir_events = dir_ad_hoc + '/events'
dir_external_val_data = dir_ad_hoc + '/external_validation_data_db'
dir_aurora_event_base = '/u/aurora-r0/cmarshak/dist-s1-research/marshak/10_ad_hoc_data_generation/out'

event_name = 'chile_fire_2024'

yaml_file = dir_events + '/' + event_name + '.yml'
with open(yaml_file) as f:
  event_dict = yaml.safe_load(f)["event"]

print(f'Analyzing Event: {event_name}')

dir_event = dir_aurora_event_base + '/' + event_name
dir_rtc = dir_event + '/rtc_ts_merged'

filename_event = dir_external_val_data + '/' + event_name + '.geojson'
filename_event_val = event_name + '.png'
event = gpd.read_file(filename_event)
fig,ax = plt.subplots()
event.plot(ax=ax)
fig.tight_layout()
fig.savefig(filename_event_val,dpi=300,bbox_inches="tight")
plt.close(fig)

# Alg parameters
td_lookback = timedelta(days=18)
td_halfwindow = timedelta(days=18)

# Tracks available for this event
tracknums = [d[5:] for d in os.listdir(dir_rtc)]

prs_dist1ds = Presentation()
#blank_slide_layout_dist1ds = prs_dist1ds.slide_layouts[6]
prs_data = Presentation()
#blank_slide_layout_data = prs_data.slide_layouts[6]
prs_dist2ds = Presentation()

do_data_plot = False
do_dist1ds_plot = False
do_dist2ds_plot = False
redo_mahalanobis = True

file_mahalanobis_base = 'mahalanobis'

try:
  #for tracknum in tracknums:
  for tracknum in [tracknums[0]]:
    print(f"Track: {tracknum}")
    file_mahalanobis_track = file_mahalanobis_base + '_' + tracknum + '.pkl'
    dir_track = dir_rtc + '/track' + tracknum
    datetimes = []
    vv_filelist = []
    vh_filelist = []
    list_dir = sorted(os.listdir(dir_track))
    ipair = 0
    for filename_rtc in list_dir:
      # Accumulate datetimes and channel filenames
      # Assuming VV and VH pairs for each datetime
      str_channel = filename_rtc[-6:-4]
      str_datetime = filename_rtc[7:17]
      if str_channel == "VV":
        vv_filelist.append(filename_rtc)
      elif str_channel == "VH":
        vh_filelist.append(filename_rtc)
      else:
        print(f'Error: invalid str_channel {str_channel}')
        raise ValueError('Bad str_channel')
      if ipair == 0:
        # New datetime
        datetime1 = pd.to_datetime(str_datetime)
        datetimes.append(datetime1)
        ipair = 1
        last_str_datetime = str_datetime
      else:
        # Already have first member of pair
        ipair = 0
        if last_str_datetime != str_datetime:
          print(f'Error: mismatched datetime {str_datetime}')
          raise ValueError('Bad pair datetime')

    # Determine previous data list within available data times

    Ndatetimes = len(datetimes)
    pre_idxs = [alg_fcns.lookup_pre_idx_daterange(dt,datetimes,
      td_halfwindow,td_lookback) for i,dt in enumerate(datetimes)]

    # Load all RTC data for available datetimes

    print("Loading RTC data")
    vv_data = [get_rtc_fromfile(os.path.join(dir_track,fname))
      for fname in vv_filelist]
    vh_data = [get_rtc_fromfile(os.path.join(dir_track,fname))
      for fname in vh_filelist]

    # Compute ratio
    vv_vh_ratio = [vv/vh for vv,vh in zip(vv_data,vh_data)]

    if (not redo_mahalanobis) or Path(file_mahalanobis_track).is_file():
      print("Loading already computed Mahalanobis distances")
      with open(file_mahalanobis_track) as fpkl:
        (vv_dist_objs,vh_dist_objs,dist1d_ratio_objs,
         dist2d_objs) = pickle.load(fpkl)
    else:
      # Compute Mahalonobis distances from RTC data using pre_idxs for
      # each datetime

      print("Computing Mahalonobis distances")

      # Collect pre arrays as views to avoid unnecessary copying
      pre_vv = [[vv_data[i].view() for i in prei] for prei in pre_idxs]
      pre_vh = [[vh_data[i].view() for i in prei] for prei in pre_idxs]
      pre_ratio = [[vv_vh_ratio[i].view() for i in prei] for prei in pre_idxs]

      vv_dist_objs = [compute_mahalonobis_dist_1d(prevv, vv)
        for prevv,vv in zip(pre_vv,vv_data)]
      vh_dist_objs = [compute_mahalonobis_dist_1d(prevh, vh)
        for prevh,vh in zip(pre_vh,vh_data)]
      dist1d_ratio_objs = [compute_mahalonobis_dist_1d(preratio,ratio)
        for preratio,ratio in zip(pre_ratio,vv_vh_ratio)]
      dist2d_objs = [compute_mahalonobis_dist_2d(prevv,prevh,vv,vh)
        for prevv,prevh,vv,vh in zip(pre_vv,pre_vh,vv_data,vh_data)]

      print("Saving Mahalanobis distances")
      with open(file_mahalanobis_track,'wb') as fpkl:
        pickle.dump([vv_dist_objs,vh_dist_objs,dist1d_ratio_objs,
          dist2d_objs],fpkl)

    # Extract distances and max/mins
    print("Extracting distances and max/mins")

    vv_data_max = [vv.max() for vv in vv_data]
    max_vv_data = max(vv_data_max)
    vv_data_min = [vv.min() for vv in vv_data]
    min_vv_data = min(vv_data_min)

    vh_data_max = [vh.max() for vh in vh_data]
    max_vh_data = max(vh_data_max)
    vh_data_min = [vh.min() for vh in vh_data]
    min_vh_data = min(vh_data_min)

    vv_dist1ds = [vv_dist_objs[i].dist for i in range(Ndatetimes)]
    vv_dist1ds_max = [vv_dist1ds[i].max() for i in range(Ndatetimes)]
    max_vv_dist1ds = max(vv_dist1ds_max)
    vv_dist1ds_min = [vv_dist1ds[i].min() for i in range(Ndatetimes)]
    min_vv_dist1ds = min(vv_dist1ds_min)
    
    vh_dist1ds = [vh_dist_objs[i].dist for i in range(Ndatetimes)]
    vh_dist1ds_max = [vh_dist1ds[i].max() for i in range(Ndatetimes)]
    max_vh_dist1ds = max(vh_dist1ds_max)
    vh_dist1ds_min = [vh_dist1ds[i].min() for i in range(Ndatetimes)]
    min_vh_dist1ds = min(vh_dist1ds_min)
    
    ratio_dist1ds = [dobj.dist for dobj in dist1d_ratio_objs]
    ratio_dist1ds_max = [np.nanmax(d1) for d1 in ratio_dist1ds]
    max_ratio_dist1ds = np.nanmax(ratio_dist1ds_max)
    ratio_dist1ds_min = [np.nanmin(d1) for d1 in ratio_dist1ds]
    min_ratio_dist1ds = np.nanmin(ratio_dist1ds_min)

    dist2ds = [dobj.dist for dobj in dist2d_objs]
    dist2ds_max = [np.nanmax(d1) for d1 in dist2ds]
    max_dist2ds = np.nanmax(dist2ds_max)
    dist2ds_min = [np.nanmin(d1) for d1 in dist2ds]
    min_dist2ds = np.nanmin(dist2ds_min)

    print("Doing powerpoint figures")
    #figfile = 't' + tracknum + '_vv_dist1ds.png'
    figfile = 'tmp.png'
    for i in range(Ndatetimes):
      print(f"{i+1}/{Ndatetimes}",end='\r')
      if i > 0:
        continue
      if do_dist1ds_plot:
        prs_implot2('VV',vv_dist1ds[i],min_vv_dist1ds,max_vv_dist1ds,
          'VH',vh_dist1ds[i],min_vh_dist1ds,max_vh_dist1ds,
          datetimes[i],tracknum,event_dict['event_date'],
          prs_dist1ds,figfile)

      if do_dist2ds_plot:
        prs_implot2('VV/VH',ratio_dist1ds[i],min_ratio_dist1ds,
          max_ratio_dist1ds,
          '2D VV,VH',dist2ds[i],min_dist2ds,max_dist2ds,
          datetimes[i],tracknum,event_dict['event_date'],
          prs_dist2ds,figfile)

      if do_data_plot:
        prs_implot2('VV',vv_data[i],0.0,0.5,
          'VH',vh_data[i],0.0,0.05,
          datetimes[i],tracknum,event_dict['event_date'],
          prs_data,figfile)

    print("\ndone")

except Exception as e:
    exc_type,exc_value,exc_traceback = sys.exc_info()
    tb = exc_traceback
    stack = inspect.trace()
    lcls = locals()
    slcls = tb.tb_next.tb_frame.f_locals
    print(f"Error at tracknum: {tracknum}, filename_rtc: {filename_rtc}")
    print(exc_value)
    traceback.print_tb(tb)

if do_dist1ds_plot:
  prs_dist1ds.save('chile_fire_dist1ds.pptx')  

if do_dist2ds_plot:
  prs_dist2ds.save('chile_fire_dist2ds.pptx')  

if do_data_plot:
  prs_data.save('chile_fire_data.pptx')  

