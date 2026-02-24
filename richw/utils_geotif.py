import sys
import os
import inspect
from typing import Tuple
from pathlib import Path
import math
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box,  Polygon
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform,reproject,Resampling
from rasterio.transform import xy
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
import pyproj
from pyproj import Transformer
#from pyproj import CRS
from pyproj import Geod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Circle
import rasterio.plot

#import dist_s1
#from dist_s1.packaging import (
#  convert_geotiff_to_png
#)

# Make image plot with lat/lon grid
def geotif_to_png_map(gtif_name,png_name,geo_crs,sinu_crs,titlestr,
    cmap_in='gray',vmin=0,vmax=0,gdf=None):
  # still need to move cmap uint8 handling in here...
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
      return extent

# Alternative using reprojection to geographic coordinates
def geotif_to_png_map2(gtif_name,png_name,geo_crs,titlestr,
    cmap_in='gray',vmin=0,vmax=0,lat1=0,lon1=0,circ_size=0.01,gdf=None):
  with rasterio.open(gtif_name) as src:
    # --- 1. Compute target transform and size in geo_crs ---
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, geo_crs,
        src.width, src.height,
        *src.bounds
    )

    # 2. Allocate destination array for one band
    dst_data = np.empty((dst_height, dst_width), dtype=src.dtypes[0])

    # --- 2. Prepare destination metadata ---
    #dst_meta = src.meta.copy()
    #dst_meta.update({
    #    "crs": geo_crs,
    #    "transform": dst_transform,
    #    "width": dst_width,
    #    "height": dst_height
    #})

    # --- 3. reproject first band ---
    reproject(
        source=src.read(1),
        destination=dst_data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=geo_crs,
        resampling=Resampling.nearest
    )

  # --- 4. Plot directly in geographic coords ---
  # Bounds in lon/lat come from dst_transform and size
  left = dst_transform.c
  top = dst_transform.f
  right = left + dst_transform.a * dst_width
  bottom = top + dst_transform.e * dst_height

  extent = [left, right, bottom, top]
  ang_size = left - right
  circ_size_deg = circ_size*ang_size

  fig, ax = plt.subplots(figsize=(8, 8), dpi = 100)

  if isinstance(cmap_in, dict) and (dst_data.dtype == np.uint8):
      # Convert to 0-1 floats for matplotlib
      default_gray = [128,128,128,255]
      colors = []
      for i in range(256):
          r,g,b,a = cmap_in.get(i,default_gray)
          colors.append([r/255.0,g/255.0,b/255.0,a/255.0])
      cmap = ListedColormap(colors)
      norm = BoundaryNorm(boundaries=np.arange(-0.5,256.5),ncolors=256)
      im = ax.imshow(
          dst_data,
          extent=extent,
          origin="upper",
          cmap=cmap,
          norm=norm,
      )
  else:
      im = ax.imshow(
          dst_data,
          extent=extent,
          origin="upper",
          cmap=cmap_in,
          vmin=vmin,
          vmax=vmax,
      )

  # Add a circle if lat1,lon1 point specified
  if lat1 is not None and lon1 is not None:
      circle = Circle(
          (lon1, lat1),           # center in lon/lat
          radius=circ_size_deg,   # radius in degrees
          edgecolor="red",
          facecolor="none",
          linewidth=1
      )
      ax.add_patch(circle)

  ax.set_xlabel("E. Longitude (deg)")
  ax.set_ylabel("Latitude (deg)")
  ax.set_title("Reprojected to EPSG:4326")
  #plt.colorbar(im, ax=ax, label="Value")
  ax.set_title(titlestr)
  plt.savefig(png_name,dpi=300,bbox_inches='tight')
  return extent


def geotif_to_png(
    gtif_name: Path,
    png_name: Path,
    cmap_in,
    vmin,
    vmax,
    row1=-1,
    col1=-1
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
        
    if row1 >= 0 and col1 >= 0:
        circle = Circle((col1,row1),
            radius=5,
            edgecolor = 'red',
            facecolor = 'none',
            linewidth = 1.5)
        ax.add_patch(circle)

    #if cmap_in_dict is None:
    #    cmap_in_dict = src.colormap(1) if src.count == 1 else None

    plt.savefig(png_name,dpi=300,bbox_inches='tight')
    #im_uint8 = Image.fromarray(arr1_uint8)
    #im_uint8.save(png_name)

def geotif_to_png_overlay(gtif_path,png_path,cmap,vmin,vmax,row1=-1,col1=-1):
    geotif_to_png(gtif_path,png_path,cmap,vmin,vmax,row1,col1)
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

    # Reduce out_arr from 3D Band intereaved by pixel (BIP) to 2D
    # and save as geotiff using out_profile
    out_arr = out_arr[0, ...]
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(out_arr, 1)

def reproject_to_geom(
    src_path: str,
    dst_crs: str,
    dst_geom: box,
    dst_res: Tuple[float, float] | None,
    dst_path: str,
    resampling: Resampling = Resampling.nearest,
) -> Tuple[float,float]:
    """
    Reproject a single raster to dst_crs and crop it to the bounding box of dst_geom.

    Parameters
    ----------
    src_path : str
        Input GeoTIFF path.
    dst_crs : str
        Destination CRS (e.g. 'EPSG:4326').
    dst_geom : shapely geometry (e.g. box, Polygon)
        Geometry in dst_crs; its bounding box defines the target area.
    dst_res : (x_res, y_res) or None
        Desired output pixel size in destination CRS units.
        If None, the resolution is chosen automatically.
    dst_path : str
        Output geotif path for reprojected/cropped raster.
    """

    # Get (left, bottom, right, top) from the Shapely geometry
    left, bottom, right, top = dst_geom.bounds
    dst_bounds = (left, bottom, right, top)

    with rasterio.open(src_path) as src:
        if dst_res is not None:
            x_res, y_res = dst_res
            dst_width = int(round((right - left) / x_res))
            dst_height = int(round((top - bottom) / abs(y_res)))
            dst_transform = rasterio.transform.from_bounds(
                left, bottom, right, top, dst_width, dst_height
            )
        else:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *dst_bounds,
            )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    return dst_width, dst_height

def meters_per_degree_lat(lat_rad: float) -> float:
    """
    Meters per degree of latitude at latitude lat_rad (radians),
    using WGS84 ellipsoid.
    """
    # Ellipsoid constants
    geod = Geod(ellps="WGS84")
    A = geod.a            # semi-major axis
    B = geod.b            # semi-minor axis
    E2 = (A**2 - B**2) / A**2  # first eccentricity squared

    # Radius of curvature in the meridian (M)
    sin_phi = math.sin(lat_rad)
    denom = (1 - E2 * sin_phi**2)
    M = (A * (1 - E2)) / (denom ** 1.5)

    # 1 degree = pi / 180 radians
    return M * (math.pi / 180.0)


def meters_per_degree_lon(lat_rad: float) -> float:
    """
    Meters per degree of longitude at latitude lat_rad (radians),
    using WGS84 ellipsoid.[web:49]
    """
    # Ellipsoid constants
    geod = Geod(ellps="WGS84")
    A = geod.a            # semi-major axis
    B = geod.b            # semi-minor axis
    E2 = (A**2 - B**2) / A**2  # first eccentricity squared

    # Radius of curvature in the prime vertical (N)
    sin_phi = math.sin(lat_rad)
    N = A / math.sqrt(1 - E2 * sin_phi**2)

    # Arc length at this latitude is N * cos(phi)
    return N * math.cos(lat_rad) * (math.pi / 180.0)


def projected_res_to_deg_res(
    src_path: str,
    dst_lat_deg: float | None = None,
) -> Tuple[float, float]:
    """
    Convert projected pixel resolution (meters) from a GeoTIFF to
    approximate angular resolution (degrees) for EPSG:4326 grids.
    """
    with rasterio.open(src_path) as src:
        x_res_m = src.transform.a
        y_res_m = -src.transform.e

        if dst_lat_deg is None:
            from rasterio.warp import transform

            cx = src.bounds.left + (src.width * src.transform.a) / 2.0
            cy = src.bounds.top + (src.height * src.transform.e) / 2.0
            (lon_deg,), (lat_deg,) = transform(src.crs, "EPSG:4326", [cx], [cy])
            dst_lat_deg = lat_deg

    lat_rad = math.radians(dst_lat_deg)

    m_per_deg_lat = meters_per_degree_lat(lat_rad)
    m_per_deg_lon = meters_per_degree_lon(lat_rad)

    x_res_deg = x_res_m / m_per_deg_lon
    y_res_deg = y_res_m / m_per_deg_lat

    return x_res_deg, y_res_deg
