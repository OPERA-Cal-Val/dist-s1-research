#! /usr/bin/env python

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import data_fcns
import warnings
from datetime import datetime, timedelta
from dateutil.parser import parse
import contextily as ctx
import numpy as np

def plot_rtc(df_rtc_ts_wind,figfile,df_site):
  POL_RATIO_PLOT = False
  BURST_ID = df_rtc_ts_wind['burst_id'].iloc[0]
  SITE_ID = df_rtc_ts_wind['site_id'].iloc[0]
  fig, ax1 = plt.subplots(figsize=(10, 5))
  ax1.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vv_avg'], marker='o', color='tab:blue', label='vv_avg')
 
  ax1.set_xlabel('Datetime')
  ax1.set_ylabel('vv_avg', color='tab:blue')
  ax1.tick_params(axis='y', labelcolor='tab:blue')
  ax2 = ax1.twinx()
  ax2.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vh_avg'], marker='v', color='tab:brown', label='vh_avg')
  ax2.set_ylabel('vh_avg', color='tab:brown')
  ax2.tick_params(axis='y', labelcolor='tab:brown')
  if POL_RATIO_PLOT:
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.plot(df_rtc_ts_wind['datetime'], df_rtc_ts_wind['vv/vh_avg'], marker='P', color='tab:purple', label='vv/vh_avg')
    ax3.set_ylabel('vv/vh_avg', color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='tab:purple')

  change_type = df_site.change_type.iloc[0]
  plt.title(f'Change type {change_type}; {BURST_ID=}; {SITE_ID=}')
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    ax1.set_xticks(df_rtc_ts_wind['datetime'].tolist())
    ax1.set_xticklabels(df_rtc_ts_wind['datetime'].tolist(), rotation=90)

  # Subsample the xticks by N
  N = 2
  ticks = plt.gca().get_xticks()
  new_ticks = ticks[::N]
  new_tick_labels = [str(int(tick)) for tick in ticks[::N]]
  plt.gca().set_xticks(new_ticks)
  plt.gca().set_xticklabels(new_tick_labels)

  ax1.grid(True)

  last_observed_time = df_site['last_observation_time'][0]

  if not pd.isnull(last_observed_time):
    ax1.axvline(x=last_observed_time, color='b', linestyle='--', label=f'Last observation time ({last_observed_time})')
  ax2.legend(loc='upper left')

  change_time = df_site['change_time'][0]
  if not pd.isnull(change_time):
    ax1.axvline(x=change_time, color='r', linestyle='--', label=f'Change time ({change_time})')

  ax1.legend()
  fig.savefig(figfile,dpi=300,bbox_inches="tight")
  plt.close(fig)

def prs_implot2(chanstr1,arr1,vmin1,vmax1,chanstr2,arr2,vmin2,vmax2,
  dt,tracknum,event_date,prs,tmpname):
  fig,ax = plt.subplots()
  im = ax.imshow(arr1,cmap='gray',vmax=vmax1,vmin=vmin1)
  plt.title(f'{chanstr1}, trk: {tracknum}, {event_date}, im date: {datetime.strftime(dt,'%y-%m-%d')}')
  fig.tight_layout()
  fig.savefig(tmpname,dpi=300,bbox_inches="tight")
  plt.close(fig)
  slide = prs.slides.add_slide(prs.slide_layouts[6])
  left = Inches(0.5)
  top = Inches(1)
  height = Inches(4)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)

  fig,ax = plt.subplots()
  im = ax.imshow(arr2,cmap='gray',vmax=vmax2,vmin=vmin2)
  plt.title(f'{chanstr2}, trk: {tracknum}, {event_date}, im date: {datetime.strftime(dt,'%y-%m-%d')}')
  fig.tight_layout()
  fig.savefig(tmpname,dpi=300,bbox_inches="tight")
  plt.close(fig)
  left = Inches(5)
  top = Inches(1)
  height = Inches(4)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)

def implot1(fig,ax,chanstr1,arr1,vmin1,vmax1,
  dt,tracknum,event_date,plotname):
  im = ax.imshow(arr1,cmap='gray',vmax=vmax1,vmin=vmin1)
  plt.title(f'{chanstr1}, trk: {tracknum}, {event_date}, im date: {datetime.strftime(dt,'%y-%m-%d')}')
  fig.tight_layout()
  fig.savefig(plotname,dpi=300,bbox_inches="tight")

def prs_roc4(chanstr1,tp1,fp1,chanstr2,tp2,fp2,chanstr3,tp3,fp3,
  chanstr4,tp4,fp4,dt,tracknum,event_date,prs,tmpname):
  slide = prs.slides.add_slide(prs.slide_layouts[6])
  roc1(chanstr1,tp1,fp1,tmpname,dt,tracknum,event_date)
  left = Inches(0.5)
  top = Inches(1)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  roc1(chanstr2,tp2,fp2,tmpname,dt,tracknum,event_date)
  left = Inches(5)
  top = Inches(1)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  roc1(chanstr3,tp3,fp3,tmpname,dt,tracknum,event_date)
  left = Inches(0.5)
  top = Inches(4)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  roc1(chanstr4,tp4,fp4,tmpname,dt,tracknum,event_date)
  left = Inches(5)
  top = Inches(4)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)

def roc1(fig,ax,chanstr,tp1,fp1,tmpname,dt,tracknum,event_date): 
  line1 = ax.plot(fp1,tp1,marker='o')
  ax.set_xlabel('false positive')
  ax.set_ylabel('true positive')
  plt.title(f'{chanstr} trk: {tracknum}, {event_date}, {datetime.strftime(dt,'%y-%m-%d')}')
  fig.savefig(tmpname,dpi=300,bbox_inches="tight")
  
def hist1(fig,ax,chanstr,arr1,arr_name,nbins,binrange,
  tmpname,dt,tracknum,event_date): 
  ax.hist(arr1,bins=nbins,range=binrange)
  ax.set_xlabel(arr_name)
  ax.set_ylabel('count')
  plt.title(f'{chanstr} trk: {tracknum}, {event_date}, {datetime.strftime(dt,'%y-%m-%d')}')
  fig.savefig(tmpname,dpi=300,bbox_inches="tight")
  
def prs_dathist2(chanstr1,data1,name1,vmin1,vmax1,nbins1,binrange1,
  chanstr2,data2,name2,vmin2,vmax2,nbins2,binrange2,
  dt,tracknum,event_date,prs,tmpname):
  if binrange1 == 0:
    binrange1 = (data1.min(),data1.max())
  if binrange2 == 0:
    binrange2 = (data2.min(),data2.max())
  slide = prs.slides.add_slide(prs.slide_layouts[6])
  fig,ax = plt.subplots()
  implot1(fig,ax,chanstr1,data1,vmin1,vmax1,dt,tracknum,event_date,tmpname)
  left = Inches(0.5)
  top = Inches(1)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  implot1(fig,ax,chanstr2,data2,vmin2,vmax2,dt,tracknum,event_date,tmpname)
  left = Inches(5)
  top = Inches(1)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  #plt.close(fig)

  #fig,ax = plt.subplots()
  ax.clear()
  hist1(fig,ax,chanstr1,data1,name1,nbins1,binrange1,
    tmpname,dt,tracknum,event_date)
  left = Inches(0.5)
  top = Inches(4)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  hist1(fig,ax,chanstr2,data2,name2,nbins2,binrange2,
    tmpname,dt,tracknum,event_date)
  left = Inches(5)
  top = Inches(4)
  height = Inches(3.5)
  width = Inches(4)
  pic = slide.shapes.add_picture(tmpname,left,top,width=width,height=height)
  plt.close(fig)

def plot_mgrs(fname,df_mgrs1,df_point,bbox):
  mgrs_tile_id = df_mgrs1['mgrs_tile_id'][0]
  df_mgrs1_plot = df_mgrs1.copy()
  df_mgrs1_plot.geometry = df_mgrs1_plot.geometry.boundary
  fig,ax = plt.subplots()
  df_mgrs1_plot.plot(column="mgrs_tile_id", categorical=True, legend=True, ax=ax)
  df_point.plot(ax=ax)
  x,y = bbox.exterior.xy
  plt.plot(x,y,color='green')
  ctx.add_basemap(ax,crs=df_mgrs1_plot.crs.to_string())
  plt.title(f'MGRS {mgrs_tile_id}')
  fig.tight_layout()
  fig.savefig(fname,dpi=300,bbox_inches="tight")

def plot_mgrs_bursts(fname,df_mgrs1,df_point,bbox,df_bursts):
  mgrs_tile_id = df_mgrs1['mgrs_tile_id'][0]
  df_mgrs1_plot = df_mgrs1.copy()
  df_mgrs1_plot.geometry = df_mgrs1_plot.geometry.boundary
  fig,ax = plt.subplots()
  df_mgrs1_plot.plot(column="mgrs_tile_id", categorical=True, legend=True, ax=ax)
  if df_point is not None:
      df_point.plot(ax=ax)
  if bbox is not None:
      x,y = bbox.exterior.xy
      plt.plot(x,y,color='green')

  df_bursts_plot = df_bursts.copy()
  df_bursts_plot.geometry = df_bursts_plot.geometry.boundary
  df_bursts_plot.plot(column="track_number",categorical=True,ax=ax,legend=True)
  leg = ax.get_legend()
  leg.set_title("Track Numbers")
  ctx.add_basemap(ax,crs=df_mgrs1_plot.crs.to_string())
  plt.title(f'MGRS {mgrs_tile_id}')
  fig.tight_layout()
  fig.savefig(fname,dpi=300,bbox_inches="tight")

def hist_df(df,col,titlestr,plotpath,nbins):
  plt.clf()
  df[col].hist(bins=nbins)
  plt.xlabel(col)
  plt.ylabel('Frequency')
  plt.title(titlestr)
  plt.tight_layout()
  filepath = Path(str(plotpath) + '_hist.png')
  plt.savefig(filepath,dpi=300,bbox_inches="tight")
  return filepath
  
def implot_gdf(gdf,col_loc,col_value,titlestr,cmap,plotpath):
  plt.clf()  
  # Extract lon/lat from midpoint points (assumes CRS is geographic)
  x = gdf[col_loc].x
  y = gdf[col_loc].y

  if isinstance(cmap,dict):
      v = gdf[col_value].to_numpy()
      colors = np.array([cmap.get(int(s),(1,1,1,1)) for s in v])
      keys = sorted(cmap.keys())
      cmap_list = [cmap[k] for k in keys]
      # Build colormap and norm in data space (days)
      listed_cmap = mcolors.ListedColormap(cmap_list)
      bounds = keys + [keys[-1] + 1] # one more than last key
      norm = mcolors.BoundaryNorm(bounds,listed_cmap.N)
      sc = plt.scatter(
        x, y,
        c=colors,
        s=7,
        marker="s",
        linewidths=0
      )
      # Colorbar uses the norm and colormap in days
      ax = plt.gca()
      sm = cm.ScalarMappable(norm=norm, cmap=listed_cmap)
      sm.set_array([])  # required in some Matplotlib versions
      cbar = plt.colorbar(sm, ax=ax)
  else:
      v = gdf[col_value]
      sc = plt.scatter(
        x, y,
        c=v,
        s=5,
        cmap=cmap,
        marker="s",
        linewidths=0
      )
      ax = plt.gca()
      cbar = plt.colorbar(sc, ax=ax, label=col_value)

  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.title(titlestr)
  plt.tight_layout()
  filepath = Path(str(plotpath) + '.png')
  plt.savefig(filepath,dpi=300,bbox_inches="tight")
  return filepath

def cmap_dict(cmap_boundaries):
    rgbk_colors = [(0,0,0,255), # black
        (127,0,255,255), # purple
        (0,0,255,255),   # dark blue
        (0,128,255,255), # light blue
        (0,255,255,255), # cyan
        (0,255,128,255), # blue-green
        (0,255,0,255),   # green
        (128,255,0,255), # yellow-green
        (255,255,0,255), # yellow
        (255,128,0,255), # orange
        (255,0,0,255),   # red
        (255,0,127,255), # purple red
        (255,0,255,255)  # violet
        ]
    Nb = len(cmap_boundaries)
    Ncolors = len(rgbk_colors)
    if Nb != Ncolors + 1:
        print("Warning: length mismatch in cmap_dict")
    tw_span_cmap = {}
    for icolor in range(0,Ncolors):
        for k in range(cmap_boundaries[icolor],cmap_boundaries[icolor+1]):
            tw_span_cmap[k] = tuple(x/255 for x in rgbk_colors[icolor])

    return tw_span_cmap,rgbk_colors

def add_image_grid_slides(
    prs,
    image_paths,
    grid,
    margins=(Inches(0.5), Inches(0.5), Inches(0.5), Inches(0.5)),
    keep_aspect=True,
    slide_layout_index=6,
):
    """
    Add one or more slides to `prs` with images arranged in a grid.

    Each slide shows up to `rows * cols` images. If the number of images
    is an exact multiple of the grid size, all slides are filled.
    """
    rows, cols = grid
    left_margin, top_margin, right_margin, bottom_margin = margins

    slide_width = prs.slide_width
    slide_height = prs.slide_height  # from Presentation, not Slide.

    cell_width = (slide_width - left_margin - right_margin) / cols
    cell_height = (slide_height - top_margin - bottom_margin) / rows

    max_per_slide = rows * cols
    slides = []

    # process images in chunks of max_per_slide
    for start in range(0, len(image_paths), max_per_slide):
        chunk = image_paths[start : start + max_per_slide]

        slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])
        slides.append(slide)

        for idx, img_path in enumerate(chunk):
            r = idx // cols
            c = idx % cols

            cell_left = left_margin + c * cell_width
            cell_top = top_margin + r * cell_height

            pic = slide.shapes.add_picture(
                img_path,
                cell_left,
                cell_top,
                width=cell_width,
                height=cell_height,
            )

            if keep_aspect:
                # scale to fit cell while preserving aspect
                scale_w = cell_width / pic.width
                scale_h = cell_height / pic.height
                scale = min(scale_w, scale_h)

                new_width = int(pic.width * scale)
                new_height = int(pic.height * scale)

                pic.left = int(cell_left) + int((cell_width - new_width) / 2)
                pic.top = int(cell_top) + int((cell_height - new_height) / 2)
                pic.width = new_width
                pic.height = new_height

    return slides

def add_image_grid_slide(
    prs,
    image_paths,
    grid,
    margins=(Inches(0.5), Inches(0.5), Inches(0.5), Inches(0.5)),
    keep_aspect=True,
    slide_layout_index=6,
):
    """
    Add a slide to `prs` with images arranged in a grid.

    Parameters
    ----------
    prs : pptx.Presentation
        Existing Presentation object.
    image_paths : list of str
        List of image filenames (paths). Extra cells stay empty if there
        are fewer images than grid cells; extra images are ignored.
    grid : (rows, cols)
        Tuple like (2, 3) for 2 rows × 3 columns.
    margins : (left, top, right, bottom)
        Margins around the slide, pptx length units (e.g., Inches()).
    keep_aspect : bool
        If True, scale images to fit cell while preserving aspect ratio.
        If False, images are stretched to fill the cell.
    slide_layout_index : int
        Layout index to use for the slide (6 is usually a blank slide).
    """
    rows, cols = grid
    left_margin, top_margin, right_margin, bottom_margin = margins

    slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])

    slide_width = prs.slide_width
    slide_height = prs.slide_height  # default depends on template. [web:25]

    cell_width = (slide_width - left_margin - right_margin) / cols
    cell_height = (slide_height - top_margin - bottom_margin) / rows

    for idx, img_path in enumerate(image_paths):
        if idx >= rows * cols:
            break

        r = idx // cols
        c = idx % cols

        cell_left = left_margin + c * cell_width
        cell_top = top_margin + r * cell_height

        # First add picture sized to cell, then optionally adjust to keep aspect.
        pic = slide.shapes.add_picture(
            img_path,
            cell_left,
            cell_top,
            width=cell_width,
            height=cell_height,
        )  # [web:19][web:20]

        if keep_aspect:
            # Compute scale to fit inside cell while preserving aspect ratio.
            scale_w = cell_width / pic.width
            scale_h = cell_height / pic.height
            scale = min(scale_w, scale_h)

            new_width = int(pic.width * scale)
            new_height = int(pic.height * scale)

            # Center the image within the cell.
            pic.left = int(cell_left) + int((cell_width - new_width) / 2)
            pic.top = int(cell_top) + int((cell_height - new_height) / 2)
            pic.width = new_width
            pic.height = new_height

    return slide
