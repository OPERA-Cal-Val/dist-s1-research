import sys
import os
import inspect
from pathlib import Path
import math
import pandas as pd
import geopandas as gpd
import numpy as np
#from pyproj import CRS
from shapely.geometry import Point
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform
from rasterio.transform import xy
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap,BoundaryNorm
import rasterio.plot

#import dist_s1
#from dist_s1.packaging import (
#  convert_geotiff_to_png
#)

# Make image plot with lat/lon grid
def geotif_to_png_map(gtif_name,png_name,geo_crs,sinu_crs,titlestr,
    cmap_in='gray',vmin=0,vmax=0,gdf=None):
  with rasterio.open(gtif_name) as src:
      raster = src.read(1)
      transform = src.transform
      src_crs = src.crs
      height,width = src.height, src.width
      rows = np.array([0,0,height-1,height-1])
      cols = np.array([0,width-1,width-1,0])
      xs,ys = rasterio.transform.xy(transform,rows,cols,offset="ul")
      transformer = Transformer.from_crs(src_crs,geo_crs,
          always_xy=True)
      corner_lons,corner_lats = transformer.transform(xs,ys)
      corner_lons = ((np.array(corner_lons) + 180) % 360) - 180
      extent = [corner_lons.min(),corner_lons.max(),corner_lats.min(),
          corner_lats.max()]
      fig,ax = plt.subplots(figsize=(8,8))
      if isinstance(cmap_in, dict) and (raster.dtype == np.uint8):
        # Convert to 0-1 floats for matplotlib
        default_gray = [128,128,128,255]
        colors = []
        for i in range(256):
            r,g,b,a = cmap_in.get(i,default_gray)
            colors.append([r/255.0,g/255.0,b/255.0,a/255.0])
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=np.arange(-0.5,256.5),ncolors=256)
        plt.imshow(raster, extent=extent, origin='upper',
            cmap=cmap, norm=norm)
      else:
        plt.imshow(raster, extent=extent, origin='upper',
            cmap=cmap_in, vmin=vmin, vmax=vmax)

      if gdf is not None:
          gdf.plot(ax=ax,facecolor="none",edgecolor="red",linewidth=0.5)

      ax.grid(True, which='both', color='cyan', linewidth=0.5)
      ax.set_xlabel("E. Longitude (deg)")
      ax.set_ylabel("Latitude (deg)")
      ax.set_title(titlestr)
      plt.savefig(png_name,dpi=300,bbox_inches='tight')

def geotif_to_png(
    gtif_name: Path,
    png_name: Path,
    cmap_in,
    vmin,
    vmax
) -> None:
  with rasterio.open(gtif_name) as src:
    raster = src.read(1)
    fig,ax = plt.subplots(figsize=(8,8))
    if isinstance(cmap_in, dict) and (raster.dtype == np.uint8):
        #mindata = np.nanmin(raster)
        #maxdata = np.nanmax(raster)
        #arr1_uint8 = (255*(raster - mindata) /
        #    (maxdata-mindata)).astype(np.uint8)
        # Convert to 0-1 floats for matplotlib
        default_gray = [128,128,128,255]
        colors = []
        for i in range(256):
            r,g,b,a = cmap_in.get(i,default_gray)
            colors.append([r/255.0,g/255.0,b/255.0,a/255.0])
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=np.arange(-0.5,256.5),ncolors=256)
        #color_list = [cmap_in.get(i, (0,0,0,255)) for i in range(256)]
        #color_array = np.array(color_list)
        #arr1_use = color_array[arr1_uint8]
        #cmap = None
        #vmin = 0
        #vmax = 255
        plt.imshow(raster, cmap=cmap, norm=norm)
    else:
        plt.imshow(raster, cmap=cmap_in, vmin=vmin, vmax=vmax)
        
    #if cmap_in_dict is None:
    #    cmap_in_dict = src.colormap(1) if src.count == 1 else None

    plt.savefig(png_name,dpi=300,bbox_inches='tight')
    #im_uint8 = Image.fromarray(arr1_uint8)
    #im_uint8.save(png_name)

def geotif_to_png_overlay(gtif_path,png_path,cmap,vmin,vmax):
    geotif_to_png(gtif_path,png_path,cmap,vmin,vmax)
    #convert_geotiff_to_png(gtif_path,png_path,colormap=cmap)
    with rasterio.open(gtif_path) as prod:
        prod_width = prod.width
        prod_height = prod.height
        prod_center_col = prod_width // 2
        prod_center_row = prod_height // 2
        prod_center_x,prod_center_y = prod.xy(prod_center_row,prod_center_col)
        prod_crs = prod.crs
        bounds = prod.bounds
    
    if prod_crs.to_string() != "EPSG:4326":
        from rasterio.warp import transform
        lon, lat = transform(prod_crs, "EPSG:4326", [prod_center_x], [prod_center_y])    
        center_lat = lat[0]
        center_lon = lon[0]
    else:
        center_lat = prod_center_y
        center_lon = prod_center_x
    
    transformer = pyproj.Transformer.from_crs(prod_crs, "EPSG:4326", always_xy=True)
    sw_lon,sw_lat = transformer.transform(bounds.left, bounds.bottom)
    ne_lon,ne_lat = transformer.transform(bounds.right, bounds.top)
    png_bounds = [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    return center_lat,center_lon,png_bounds

def reproject_geotiff(src_path, target_path, out_path):
    # Open the source and target GeoTIFF files and read arrays and metadata
    with rasterio.open(src_path) as src, rasterio.open(target_path) as target:
        src_arr = src.read(1)
        src_profile = src.profile
        target_profile = target.profile

    # Reproject source array to match the target profile
    out_arr, out_profile = reproject_arr_to_match_profile(
        src_arr,
        src_profile,
        target_profile,
        resampling='bilinear'
    )

    # Reduce out_arr to 2D and save as geotiff using out_profile
    out_arr = out_arr[0, ...]
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(out_arr, 1)

