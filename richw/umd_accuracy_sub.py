#! /u/aurora-r0/richw/pkgs/miniforge3/envs/dist-s1/bin/python -i

import os
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
import distmetrics
import data_fcns
import umd_fcns


filename_burst_id = 'opera_burst_ids.geojson.zip'
url_burst_id = 'https://github.com/opera-adt/burst_db/releases/download/v0.3.1/burst-id-geometries-simple-0.3.1.geojson.zip'
filename_slcs_for_processing = 'slcs_for_processing.csv.zip'
#url_maplabels = 'https://github.com/OPERA-Cal-Val/DIST-Validation/blob/main/mapLabelsv1sample'
url_maplabels_base = 'https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/mapLabelsv1sample'
filename_rtc_s1_table = 'rtc_s1_table.json.zip'
dir_rtc_data = '/u/aurora-r0/cmarshak/dist-s1-research/marshak/6_torch_dataset/opera_rtc_data'

# Subset of sites
# Read the table made by CM
# marshak/3_dist_sites/dist_hls_validation_table.ipynb
df_slcs = pd.read_csv(filename_slcs_for_processing)
sites_used = df_slcs.site_id.unique()

# Import strata and sample unit information
url_maplabels_base = 'https://raw.githubusercontent.com/OPERA-Cal-Val/DIST-Validation/main/mapLabelsv1sample'
#mapsource = "mapLabelsv1sample"
mapsource = url_maplabels_base
ANNname = "v1sample"

sampleDict = {}
sampleDict_sub = {}
sampleFull = {}
strata = {}
with open("sampledpixels1214.csv",'r') as sample:
  lines = sample.readlines()[1:]
  for l in lines:
    (ID,Stratum,Long,Lat,zone,x,y,pixel,pline,centxUTM,centyUTM,MGRS) = l.strip().split(',')
    sampleDict[ID] = ','.join([Long,Lat,zone,centxUTM,centyUTM])
    sampleFull[ID] = l.strip()
    strata[ID] = int(Stratum)
    if int(ID) in sites_used:
      sampleDict_sub[ID] = ','.join([Long,Lat,zone,centxUTM,centyUTM])

ids_all = sampleDict.keys()
ids = sampleDict_sub.keys()

#missingBaseline = []#[3,28,87,113,138,139,154,167,205,245]
#missingBaseline_ve = []#[3,28,113,138,139,205,245]
#badQA = [41,85,147]
#refVLmaj = []#[13]

#Strata area
strataAreas = {}
strataCounts = {}
with open("stratatable_0119_z.txt",'r') as file:
  lines = file.readlines()[1:]
for l in lines:
  (s,count,area,K,zcount) = l.strip().split('\t')
  strataAreas[s] = float(area)
  strataCounts[s] = int(zcount)
Nstrata = len(strataCounts)
print(f"Nstrata = {Nstrata}")

#strata enum {CONFHIVEG=6, PROVHIVEG=5, CONFLOWVEG=4,PROVLOWVEG=3,GENDIST=2,NODIST=1,NODATA=0};
#mapsource = "mapLabelsv1sample"
mapsource = url_maplabels_base
print(f"mapsource = {mapsource}")

for cattype in ["gt50"]:#["gt50","lt50","confgt50","conflt50"]:#["gt50_onlyprov","confgt50_curr","provgt50_curr","gt50","lt50","provgt50","provlt50","gt50_onlyprov","lt50_onlyprov","confgt50","conflt50"]:
    basename = "v1sample_"+cattype
    map = umd_fcns.getDISTALERTStatus_vI(mapsource,ANNname,ids,True)
    mapsimple = umd_fcns.filterMap(ids,map,cattype)
    for duration in [15]:#,5,10,15]:
      for lookback in [30]:#1,15,30]:
        name = basename+"_lookback"+str(lookback)+"_duration"+str(duration)
        noLabels = ["OCmin","OCmaj","OCtotal","noChange","VLmin"]
        ref = umd_fcns.getRefALERTbinaryDaily(["VLmaj","VLtotal"],noLabels)
        ref1 = ref
        #with open("ref_accuracy_V1.txt","w") as OUT:
        #    OUT.write(str(ref))
        print("\n"+name)
        (n,ntotal) = umd_fcns.alertConfusionMatrix_vTS2(ids,cattype,map,ref,strata,strataCounts,duration,[],lookback,name,False)
        umd_fcns.accuracy(n,ntotal,strataCounts,name,True)
        n1 = n
        ntotal1 = ntotal
        
    for duration in [15]:#,5,10,15]:
      for lookback in [30]:#1,15,30]:
        name = basename+"_lookback"+str(lookback)+"_duration"+str(duration)+"_VLmin"
        noLabels = ["OCmin","OCmaj","OCtotal","noChange"]
        ref = umd_fcns.getRefALERTbinaryDaily(["VLmaj","VLtotal","VLmin"],noLabels)
        ref2 = ref
        print("\n"+name)
        #print(mapsimple['1'])
        #print(ref['1'])
        (n,ntotal) = umd_fcns.alertConfusionMatrix_vTS2(ids,cattype,map,ref,strata,strataCounts,duration,[1],lookback,name,False)
        umd_fcns.accuracy(n,ntotal,strataCounts,name,True)
        n2 = n
        ntotal2 = ntotal
