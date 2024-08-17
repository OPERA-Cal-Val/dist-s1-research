#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1/bin/python -i

from pptx import Presentation
from pptx.util import Inches
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
blank_slide_layout_dist1ds = prs_dist1ds.slide_layouts[6]
prs_data = Presentation()
blank_slide_layout_data = prs_data.slide_layouts[6]
do_data_plot = True
do_dist1ds_plot = False

try:
  #for tracknum in tracknums:
  for tracknum in [tracknums[0]]:
    print(f"Track: {tracknum}")
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

    pre_idxs = [alg_fcns.lookup_pre_idx_daterange(dt,datetimes,
      td_halfwindow,td_lookback) for i,dt in enumerate(datetimes)]

    # Load all RTC data for available datetimes

    print("Loading RTC data")
    vv_data = [get_rtc_fromfile(os.path.join(dir_track,fname))
      for i,fname in enumerate(vv_filelist)]
    vv_data_max = [vv_data[i].max() for i in range(len(datetimes))]
    max_vv_data = max(vv_data_max)
    vv_data_min = [vv_data[i].min() for i in range(len(datetimes))]
    min_vv_data = min(vv_data_min)

    vh_data = [get_rtc_fromfile(os.path.join(dir_track,fname))
      for i,fname in enumerate(vh_filelist)]

    # Compute Mahalonobis distances from RTC data using pre_idxs for
    # each datetime

    print("Computing Mahalonobis distances")
    vv_dist_objs = [
      compute_mahalonobis_dist_1d([vv_data[j] for j in pre_idxs[i]], vv_data[i])
      for i,dt in enumerate(datetimes)]
    vv_dist1ds = [vv_dist_objs[i].dist for i in range(len(datetimes))]
    vv_dist1ds_max = [vv_dist1ds[i].max() for i in range(len(datetimes))]
    max_vv_dist1ds = max(vv_dist1ds_max)
    vv_dist1ds_min = [vv_dist1ds[i].min() for i in range(len(datetimes))]
    min_vv_dist1ds = min(vv_dist1ds_min)
    
    vh_dist_objs = [
      compute_mahalonobis_dist_1d([vh_data[j] for j in pre_idxs[i]], vh_data[i])
      for i,dt in enumerate(datetimes)]
    vh_dist1ds = [vh_dist_objs[i].dist for i in range(len(datetimes))]
    vh_dist1ds_max = [vh_dist1ds[i].max() for i in range(len(datetimes))]
    max_vh_dist1ds = max(vh_dist1ds_max)
    vh_dist1ds_min = [vh_dist1ds[i].min() for i in range(len(datetimes))]
    min_vh_dist1ds = min(vh_dist1ds_min)
    
    Ndatetimes = len(datetimes)
    print("Doing figures")
    #figfile = 't' + tracknum + '_vv_dist1ds.png'
    figfile = 'tmp.png'
    for i,dist1ds in enumerate(vv_dist1ds):
      print(f"{i+1}/{Ndatetimes}",end='\r')
      if do_dist1ds_plot:
        fig,ax = plt.subplots()
        im = ax.imshow(dist1ds,cmap='gray',
          vmax=max_vv_dist1ds,vmin=min_vv_dist1ds)
        plt.title(f'{event_name} event date: {event_dict['event_date']}, im date: {datetime.strftime(datetimes[i],'%y-%m-%d')}')
        fig.tight_layout()
        fig.savefig(figfile,dpi=300,bbox_inches="tight")
        plt.close(fig)

        slide = prs_dist1ds.slides.add_slide(blank_slide_layout_dist1ds)
        left = Inches(0.5)
        top = Inches(1)
        height = Inches(6)
        width = Inches(9)
        pic = slide.shapes.add_picture(figfile,left,top,width=width,height=height)

      if do_data_plot:
        fig,ax = plt.subplots()
        im = ax.imshow(vv_data[i],cmap='gray',
          vmax=0.1,vmin=0.0)
        plt.title(f'{event_name} event date: {event_dict['event_date']}, im date: {datetime.strftime(datetimes[i],'%y-%m-%d')}')
        fig.tight_layout()
        fig.savefig(figfile,dpi=300,bbox_inches="tight")
        plt.close(fig)

        slide = prs_data.slides.add_slide(blank_slide_layout_data)
        left = Inches(0.5)
        top = Inches(1)
        height = Inches(6)
        width = Inches(9)
        pic = slide.shapes.add_picture(figfile,left,top,width=width,height=height)

    print("done")

except Exception as e:
    exc_type,exc_value,exc_traceback = sys.exc_info()
    tb = exc_traceback
    stack = inspect.trace()
    lcls = locals()
    slcls = tb.tb_next.tb_frame.f_locals
    print(f"Alg Error at tracknum: {tracknum}, filename_rtc: {filename_rtc}")
    print(exc_value)

if do_dist1ds_plot:
  prs_dist1ds.save('chile_fire_dist1ds.pptx')  

if do_data_plot:
  prs_data.save('chile_fire_data.pptx')  

