#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from distmetrics import compute_mahalonobis_dist_1d
from distmetrics import compute_mahalonobis_dist_2d
from distmetrics import compute_log_ratio
import data_fcns
from plot_fcns import prs_implot2,prs_roc4
from plot_fcns import roc1,hist1

# Functions to support disturbance algorithms

# From dist-s1-validation-harness/ts-explore-by-site.ipynb
def window_mahalanobis_1d_running(window_data: list[list], n_pre_img: int = 5):
    arrs = [np.array(window_l).reshape((3, 3)) for window_l in window_data]
    N = len(arrs)
    dists = [compute_mahalonobis_dist_1d(arrs[k- n_pre_img : k ], arrs[k]) if k > n_pre_img else None for k in tqdm(range(N))]
    return dists

# From dist-s1-validation-harness/ts-explore-by-site.ipynb
def window_mahalanobis_1d_workflow1(window_data: list[list], T: int = 3, n_pre_img: int = 5, n_post_imgs_to_confirm: int = 3):
    arrs = [np.array(window_l).reshape((3, 3)) for window_l in window_data]
    N = len(arrs)
    dist_objs = [compute_mahalonobis_dist_1d(arrs[i - n_pre_img : i], 
                                             [arrs[i + j] for j in range(n_post_imgs_to_confirm)],
                                             window_size=3) 
                 if (i > n_pre_img) and (i + n_pre_img - 1) < N 
                 else None 
                 for i in tqdm(range(N))]
    # 1 for change and 0 elsewhere
    # note the [1, 1] indexing!
    change_pts_init = [int(all([d[1, 1] > T for d in dist_ob.dist])) if dist_ob is not None else 0 for dist_ob in dist_objs]
    # Change
    try:
        ind_c = change_pts_init.index(1)
        change_pts = [int(ind >= ind_c) for ind in range(len(change_pts_init))]
    # No Change
    except ValueError:
        change_pts = change_pts_init
    return np.array(change_pts)

# From dist-s1-validation-harness/ts-explore-by-site-stream.ipynb
def generate_post_idxs_for_workflow(acq_dt_l: list[pd.Timestamp],
                                    n_pre_img: int, 
                                    n_post_imgs_to_confirm: int,
                                    lookback_length_days: int) -> list:
  n_acqs = len(acq_dt_l)
  if lookback_length_days >= 60:
    temporal_window_size = n_pre_img // 2 + 1
    valid_post_idxs = [i for i, ts in enumerate(acq_dt_l) if
      (ts >= acq_dt_l[temporal_window_size] +
       pd.Timedelta(lookback_length_days, 'd')) and
      (i > n_pre_img) and (i < n_acqs - n_post_imgs_to_confirm + 1)
      ]
  elif lookback_length_days == 0:
    valid_post_idxs = list(range(n_pre_img + 1, n_acqs)) 
  else:
    raise ValueError('if lookback_length_days is nonzero, must be greater than 60')
  return valid_post_idxs


def lookup_pre_idx(post_idx: int,
                   acq_dt_l: list[pd.Timestamp],
                   n_pre_img: int,
                   lookback_length_days: int):
    if lookback_length_days >= 60:
        window_size = n_pre_img // 2
        valid_pre_idx = [i for (i, ts) in enumerate(acq_dt_l) if ts < (acq_dt_l[post_idx] - pd.Timedelta(lookback_length_days, 'd'))]
        if valid_pre_idx:
            pre_idx_center = valid_pre_idx[-1]
            pre_idxs = list(range(pre_idx_center - window_size, pre_idx_center + window_size + 1))
        else:
            raise ValueError('post idx does nothave valid pre-indices')
    elif lookback_length_days == 0:
         pre_idxs = list(range(post_idx - n_pre_img - 1, post_idx))
    else:
        raise ValueError('if lookback_length_days is nonzero, must be greater than 60')
    return pre_idxs

def lookup_pre_idx_daterange(post_dt: pd.Timestamp,
                   acq_dt_l: list[pd.Timestamp],
                   td_halfwindow: timedelta,
                   td_lookback: timedelta):
  dt_ref1 = post_dt - td_lookback - td_halfwindow
  dt_ref2 = post_dt - td_lookback + td_halfwindow
  pre_idxs = data_fcns.indices(acq_dt_l,
    lambda x: (x>=dt_ref1 and x <= dt_ref2))
  return pre_idxs

def window_mahalanobis_1d_workflow2(arr_data_l: list[list], 
                                   acq_dt_l: list[pd.Timestamp],
                                   T: float = 3.0,
                                   n_pre_img: int = 5, 
                                   n_post_imgs_to_confirm: int = 3,
                                   lookback_length_days: int = 365
                                   ):
    N = len(arr_data_l)
    M = len(acq_dt_l)
    if N != M:
        raise ValueError('arr_data_l and acq_dt_l must have same length')

    if (lookback_length_days != 0) and (lookback_length_days < 60):
        raise ValueError('If using non-zero lookback_length, must be larger than 60 days')
    valid_post_idxs = generate_post_idxs_for_workflow(acq_dt_l, n_pre_img, n_post_imgs_to_confirm, lookback_length_days)
    valid_pre_idxs = [lookup_pre_idx(post_idx, acq_dt_l, n_pre_img, lookback_length_days) for post_idx in valid_post_idxs]
    
    dist_objs = [compute_mahalonobis_dist_1d(arr_data_l[pre_idxs[0]: pre_idxs[-1]], 
                                             [arr_data_l[post_idx + j] for j in range(n_post_imgs_to_confirm)],
                                             window_size=3) for (pre_idxs, post_idx) in zip(valid_pre_idxs, valid_post_idxs)]
    dist1ds = []
    for d1 in dist_objs:
      if d1 is not None:
        # Only the first of the confirming post images is used here
        dist1ds.append(d1.dist[0][1,1])
      else:
        dist1ds.append(0.0)
      
    # 1 for change and 0 elsewhere
    # note the [1, 1] indexing!
    change_pts_init = [int(all([d[1, 1] > T for d in dist_ob.dist])) if dist_ob is not None else 0 for dist_ob in dist_objs]
    # Change
    try:
        ind_c = change_pts_init.index(1)
        change_pts = [int(ind >= ind_c) for ind in range(len(change_pts_init))]
    # No Change
    except ValueError:
        change_pts = change_pts_init
    valid_dts = pd.Series([acq_dt_l[k] for k in valid_post_idxs])
    return valid_dts, np.array(change_pts), dist1ds

def window_mahalanobis_2d_workflow2(arr1_data_l: list[list], 
                                   arr2_data_l: list[list],
                                   acq_dt_l: list[pd.Timestamp],
                                   T: float = 3.0,
                                   n_pre_img: int = 5, 
                                   n_post_imgs_to_confirm: int = 3,
                                   lookback_length_days: int = 365
                                   ):
    N = len(arr1_data_l)
    M = len(acq_dt_l)
    if N != M:
        raise ValueError('arr1_data_l and acq_dt_l must have same length')

    if (lookback_length_days != 0) and (lookback_length_days < 60):
        raise ValueError('If using non-zero lookback_length, must be larger than 60 days')
    valid_post_idxs = generate_post_idxs_for_workflow(acq_dt_l, n_pre_img, n_post_imgs_to_confirm, lookback_length_days)
    valid_pre_idxs = [lookup_pre_idx(post_idx, acq_dt_l, n_pre_img, lookback_length_days) for post_idx in valid_post_idxs]
    
    dist_objs = [compute_mahalonobis_dist_2d(arr1_data_l[pre_idxs[0]: pre_idxs[-1]],arr2_data_l[pre_idxs[0]: pre_idxs[-1]], 
                                             [arr1_data_l[post_idx + j] for j in range(n_post_imgs_to_confirm)],[arr2_data_l[post_idx + j] for j in range(n_post_imgs_to_confirm)],
                                             window_size=3) for (pre_idxs, post_idx) in zip(valid_pre_idxs, valid_post_idxs)]

    # RW: distds code below added to return distances used to make change_pts
    dist1ds = []
    for d1 in dist_objs:
      if d1 is not None:
        # Only the first of the confirming post images is used here
        dist1ds.append(d1.dist[0][1,1])
      else:
        dist1ds.append(0.0)
      
    # 1 for change and 0 elsewhere
    # note the [1, 1] indexing!
    change_pts_init = [int(all([d[1, 1] > T for d in dist_ob.dist])) if dist_ob is not None else 0 for dist_ob in dist_objs]
    # Change
    try:
        ind_c = change_pts_init.index(1)
        change_pts = [int(ind >= ind_c) for ind in range(len(change_pts_init))]
    # No Change
    except ValueError:
        change_pts = change_pts_init
    valid_dts = pd.Series([acq_dt_l[k] for k in valid_post_idxs])
    # RW: added dist1ds return variable
    return valid_dts, np.array(change_pts), dist1ds

# Load one raster data array from a file
def get_raster_fromfile(filepath: str) -> list[np.ndarray]:
  with rasterio.open(filepath) as rtc_dataset:
    #window = Window(915,915,915,915)
    #band1 = rtc_dataset.read(1,window=window)
    band1 = rtc_dataset.read(1)
    return band1

def anal_2d_alg(algctrl,datetimes,tracknum,event_date,
  algstr,data1,data2,pre_idxs,post_idxs,thresholds,refs,mask):
  print("Computing 2D Mahalanobis distances")
  pre_1 = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post_1 = [[data1[i].view() for i in posti] for posti in post_idxs]
  pre_2 = [[data2[i].view() for i in prei] for prei in pre_idxs]
  post_2 = [[data2[i].view() for i in posti] for posti in post_idxs]
  dist_objs = [compute_mahalonobis_dist_2d(pre1,pre2,post1,post2)
    for pre1,pre2,post1,post2 in zip(pre_1,pre_2,post_1,post_2)]
  dist1ds,change,tp,fp,tn,fn = anal_alg(algctrl,datetimes,tracknum,event_date,
    algstr,dist_objs,post_idxs,thresholds,refs,mask) 
  return dist1ds,change,tp,fp,tn,fn

def anal_1d_alg(algctrl,datetimes,tracknum,event_date,
  algstr,data1,pre_idxs,post_idxs,thresholds,refs,mask):
  print("Computing 1D Mahalanobis distances")
  pre = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post = [[data1[i].view() for i in posti] for posti in post_idxs]
  dist_objs = [compute_mahalonobis_dist_1d(pre,post,3,True,1e-4)
    for pre,post in zip(pre,post)]
  dist1ds,change,tp,fp,tn,fn = anal_alg(algctrl,datetimes,tracknum,event_date,
    algstr,dist_objs,post_idxs,thresholds,refs,mask) 
  return dist1ds,change,tp,fp,tn,fn

def anal_alg(algctrl,datetimes,tracknum,event_date,
  algstr,dist_objs,post_idxs,thresholds,refs,mask):
  Nconfirm = algctrl['Nconfirm']
  Ndatetimes = len(datetimes)
  print("Extract distances and max/mins")
  dist1ds = [dobj.dist if dobj else
    [np.zeros_like(mask,dtype='float32') for col in range(Nconfirm)]
    for dobj in dist_objs]
  dist1ds_max = [[np.nanmax(d1) for d1 in post] for post in dist1ds]
  max_dist1ds = np.nanmax([np.nanmax(d) for d in dist1ds_max])
  dist1ds_min = [[np.nanmin(d1) for d1 in post] for post in dist1ds]
  min_dist1ds = np.nanmin([np.nanmin(d) for d in dist1ds_max])
  # Change map and Accuracy metrics iterated over datetimes and thresholds
  print("Compute changes and evaluate true/false positive/negative results")
  change = [[[d > t for t in thresholds] for d in post] for post in dist1ds]
  # cumulative AND together the post results
  print("Cumulative AND the post changes")
  #change2 = [[[np.cumprod(c) for c in ct] for ct in post] for post in change]
  print("Accuracy calculations")
  tp,fp,tn,fn = accuracy2(change,refs,Nconfirm,post_idxs,mask)
  #return dist1ds,change,tp,fp,tn,fn
  if algctrl['do_roc_plot']:
    print("Doing roc plots")
    fig,ax = plt.subplots()
    plotbase = algctrl['plot_dir'] + '/roc_' + algstr + '_' + tracknum
    plotnum = 0
    for i in range(Ndatetimes):
      if datetimes[i] < event_date:
        # Dates before event have no reference positives so skip accuracy plots
        continue
      print(f"{i+1}/{Ndatetimes}",end='\r')
      for j in range(Nconfirm):
        if j < len(tp[i]):
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          if plotnum == 0:
            line1 = ax.plot(fp[i][j],tp[i][j],marker='o')
          else:
            line1[0].set_data(fp[i][j],tp[i][j])
          plt.xlim(0.0,1.0)
          plt.ylim(0.0,1.0)
          ax.set_xlabel('false positive')
          ax.set_ylabel('true positive')
          plt.title(f'{algstr} trk: {tracknum}, {event_date}, {datetime.strftime(datetimes[i],'%y-%m-%d')}')
          fig.savefig(plotname,dpi=300,bbox_inches="tight")
          #roc1(fig,ax,algstr,tp[i][j],fp[i][j],plotname,
          #  datetimes[i],tracknum,event_date)
          plotnum += 1
    plt.close(fig)
  if algctrl['do_dist1ds_plot']:
    print("Doing dist1ds plots")
    plotbase = algctrl['plot_dir'] + '/dist1ds_' + algstr + '_' + tracknum
    for i in range(Ndatetimes):
      print(f"{i+1}/{Ndatetimes}",end='\r')
      plotname = plotbase + '_' + str(i) + '.png'
      implot1(algstr,tp[i],fp[i],plotname,datetimes[i],tracknum,event_date)
  if algctrl['do_hist_mahalanobis_plot']:
    print("Doing Mahalanobis histogram plots")
    arrname = 'Mahalanobis Distance'
    plotbase = algctrl['plot_dir'] + '/mhist_' + algstr + '_' + tracknum
    for i in range(Ndatetimes):
      print(f"{i+1}/{Ndatetimes}",end='\r')
      for j in range(Nconfirm):
        if j < len(dist1ds[i]):
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          hist1(algstr,dist1ds[i][j],arrname,plotname,
            datetimes[i],tracknum,event_date)
  return dist1ds,change,tp,fp,tn,fn

def anal_1d_logratio_aoi(algctrl,datetimes,tracknum,event_date,
  algstr,data1,pre_idxs,post_idxs,ref,mask):
  print("Computing 1D Mahalanobis distances")
  pre = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post = [[data1[i].view() for i in posti] for posti in post_idxs]
  metric = [[-(np.clip(compute_log_ratio(pre1,posti),-10,-1e-5))
    for posti in post1]
    for pre1,post1 in zip(pre,post)]
  thresholds = anal_alg_aoi(algctrl,datetimes,tracknum,
    event_date,algstr,metric,post_idxs,ref,mask) 
  return thresholds

def anal_1d_mahalanobis_aoi(algctrl,datetimes,tracknum,event_date,
  algstr,data1,pre_idxs,post_idxs,ref,mask):
  Nconfirm = algctrl['Nconfirm']
  print("Computing 1D Mahalanobis distances")
  zmatrix = np.zeros_like(mask,dtype='float32')
  pre = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post = [[data1[i].view() for i in posti] for posti in post_idxs]
  dist_objs = [compute_mahalonobis_dist_1d(pre1,post1,3,True,1e-4)
    for pre1,post1 in zip(pre,post)]
  metrics = [dobj.dist if dobj else
    [zmatrix.view() for col in range(Nconfirm)]
    for dobj in dist_objs]
  thresholds = anal_alg_aoi(algctrl,datetimes,tracknum,
    event_date,algstr,metrics,post_idxs,ref,mask) 
  return thresholds

def anal_2d_mahalanobis_aoi(algctrl,datetimes,tracknum,event_date,
  algstr,data1,data2,pre_idxs,post_idxs,ref,mask):
  Nconfirm = algctrl['Nconfirm']
  print("Computing 2D Mahalanobis distances")
  zmatrix = np.zeros_like(mask,dtype='float32')
  pre_1 = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post_1 = [[data1[i].view() for i in posti] for posti in post_idxs]
  pre_2 = [[data2[i].view() for i in prei] for prei in pre_idxs]
  post_2 = [[data2[i].view() for i in posti] for posti in post_idxs]
  dist_objs = [compute_mahalonobis_dist_2d(pre1,pre2,post1,post2)
    for pre1,pre2,post1,post2 in zip(pre_1,pre_2,post_1,post_2)]
  metrics = [dobj.dist if dobj else
    [zmatrix.view() for col in range(Nconfirm)]
    for dobj in dist_objs]
  thresholds = anal_alg_aoi(algctrl,datetimes,tracknum,
    event_date,algstr,metrics,post_idxs,ref,mask) 
  return thresholds

def anal_alg_aoi(algctrl,datetimes,tracknum,event_date,
  algstr,metrics,post_idxs,ref,mask):
  Nconfirm = algctrl['Nconfirm']
  Ndatetimes = len(datetimes)
  print(f"{algstr}: Evaluate performance using sklearn.metrics")
  if algctrl['do_roc_plot']:
    print("Doing roc plots")
    fig,ax = plt.subplots()
    plotbase = algctrl['plot_dir'] + '/roc_' + algstr + '_' + tracknum
    plotnum = 0
    for i in range(Ndatetimes):
      if not metrics[i]:
        # No data available so skip this time
        continue
      if datetimes[i] < event_date:
        # Dates before event have no reference positives so skip accuracy plots
        continue
      print(f"{i+1}/{Ndatetimes}",end='\r')
      for j in range(Nconfirm):
        if j < len(metrics[i]):
          metric = metrics[i][j]
          # Restrict changes to AOI mask and exlcude NAN's
          # Apply same mask to ref data
          aoi_mask = mask & (~np.isnan(metric))
          fpr,tpr,thresholds = roc_curve(ref[aoi_mask],metric[aoi_mask])
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          if plotnum == 0:
            line1 = ax.plot(fpr,tpr)
          else:
            line1[0].set_data(fpr,tpr)
          plt.xlim(0.0,1.0)
          plt.ylim(0.0,1.0)
          ax.set_xlabel('false positive')
          ax.set_ylabel('true positive')
          plt.title(f'{algstr} trk: {tracknum}, {event_date}, {datetime.strftime(datetimes[i],'%y-%m-%d')}')
          fig.savefig(plotname,dpi=300,bbox_inches="tight")
          plotnum += 1
    plt.close(fig)
  if algctrl['do_prc_plot']:
    print("Doing prc plots")
    fig,ax = plt.subplots()
    plotbase = algctrl['plot_dir'] + '/prc_' + algstr + '_' + tracknum
    plotnum = 0
    for i in range(Ndatetimes):
      if not metrics[i]:
        # No data available so skip this time
        continue
      if datetimes[i] < event_date:
        # Dates before event have no reference positives so skip accuracy plots
        continue
      print(f"{i+1}/{Ndatetimes}",end='\r')
      for j in range(Nconfirm):
        if j < len(metrics[i]):
          metric = metrics[i][j]
          # Restrict changes to AOI mask and exlcude NAN's
          # Apply same mask to ref data
          aoi_mask = mask & (~np.isnan(metric))
          precision,recall,thresholds = precision_recall_curve(
            ref[aoi_mask],metric[aoi_mask])
          #precision,recall,f1,support = precision_recall_fscore_support(
          #  ref[aoi_mask],metric[aoi_mask])
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          if plotnum == 0:
            plt.cla()
            line1 = ax.plot(recall,precision)
          else:
            line1[0].set_data(fpr,tpr)
          plt.xlim(0.0,1.0)
          plt.ylim(0.0,1.0)
          ax.set_xlabel('recall')
          ax.set_ylabel('precision')
          plt.title(f'{algstr} trk: {tracknum}, {event_date}, {datetime.strftime(datetimes[i],'%y-%m-%d')}')
          fig.savefig(plotname,dpi=300,bbox_inches="tight")
          #plotnum += 1
    plt.close(fig)
  if algctrl['do_metric_plot']:
    print("Doing metric image")
    fig,ax = plt.subplots()
    vmin1 = algctrl['metric_plot_vmin']
    vmax1 = algctrl['metric_plot_vmax']
    plotbase = algctrl['plot_dir'] + '/metric_' + algstr + '_' + tracknum
    for i in range(Ndatetimes):
      if not metrics[i]:
        # No data available so skip this time
        continue
      if datetimes[i] < event_date:
        # Dates before event have no reference positives so skip accuracy plots
        continue
      print(f"{i+1}/{Ndatetimes}",end='\r')
      for j in range(Nconfirm):
        if j < len(metrics[i]):
          metric = metrics[i][j]
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          plt.cla()
          im1 = ax.imshow(metric,cmap='gray',vmax=vmax1,vmin=vmin1)
          plt.title(f'{algstr} trk: {tracknum}, {event_date}, {datetime.strftime(datetimes[i],'%y-%m-%d')}')
          fig.tight_layout()
          fig.savefig(plotname,dpi=300,bbox_inches="tight")
    plt.close(fig)
  print("plots done")

  return thresholds

def anal_1d_alg_aoi2(algctrl,datetimes,tracknum,event_date,
  algstr,data1,pre_idxs,post_idxs,thresholds,ref,mask):
  print("Computing 1D Mahalanobis distances")
  pre = [[data1[i].view() for i in prei] for prei in pre_idxs]
  post = [[data1[i].view() for i in posti] for posti in post_idxs]
  dist_objs = [compute_mahalonobis_dist_1d(pre,post,3,True,1e-4)
    for pre,post in zip(pre,post)]
  dist1ds,change,tp,fp,tn,fn = anal_alg_aoi2(algctrl,datetimes,tracknum,
    event_date,algstr,dist_objs,post_idxs,thresholds,ref,mask) 
  return dist1ds,change,tp,fp,tn,fn

def anal_alg_aoi2(algctrl,datetimes,tracknum,event_date,
  algstr,dist_objs,post_idxs,thresholds,ref_aoi,mask):
  Nconfirm = algctrl['Nconfirm']
  Ndatetimes = len(datetimes)
  zmatrix = np.zeros_like(mask,dtype='float32')
  print("Extract distances and max/mins")
  dist1ds = [dobj.dist if dobj else
    [zmatrix.view() for col in range(Nconfirm)]
    for dobj in dist_objs]
  dist1ds_max = [[np.nanmax(d1) for d1 in post] for post in dist1ds]
  max_dist1ds = np.nanmax([np.nanmax(d) for d in dist1ds_max])
  dist1ds_min = [[np.nanmin(d1) for d1 in post] for post in dist1ds]
  min_dist1ds = np.nanmin([np.nanmin(d) for d in dist1ds_max])
  # Change map and Accuracy metrics iterated over datetimes and thresholds
  print("Compute changes and evaluate true/false positive/negative results")
  # Restrict changes to AOI mask
  dist1ds_aoi = [[d1[mask] for d1 in d] for d in dist1ds]
  change = [[[d > t for d in post] for t in thresholds] for post in dist1ds_aoi]
  #change = [[[d > t for t in thresholds] for d in post] for post in dist1ds]
  # cumulative AND together the post results
  print("Cumulative AND the post changes")
  change2 = [[np.cumprod(np.stack(post),axis=0)
    for post in ct] for ct in change]
  print("Accuracy calculations")
  tp,fp,tn,fn = accuracy(change2,ref_aoi,Nconfirm,post_idxs)
  #return dist1ds,change,tp,fp,tn,fn
  if algctrl['do_roc_plot']:
    print("Doing roc plots")
    fig,ax = plt.subplots()
    plotbase = algctrl['plot_dir'] + '/roc_' + algstr + '_' + tracknum
    plotnum = 0
    for i in range(Ndatetimes):
      if datetimes[i] < event_date:
        # Dates before event have no reference positives so skip accuracy plots
        continue
      print(f"{i+1}/{Ndatetimes}",end='\r')
      fpi = [list(inner) for inner in zip(*fp[i])]
      tpi = [list(inner) for inner in zip(*tp[i])]
      for j in range(Nconfirm):
        if j < len(tpi):
          plotname = plotbase + '_' + str(i) + '_' + str(j) + '.png'
          if plotnum == 0:
            line1 = ax.plot(fpi[j],tpi[j],marker='o')
          else:
            line1[0].set_data(fpi[j],tpi[j])
          #flattened_list = list(itertools.chain(*nested_list))
          plt.xlim(0.0,1.0)
          plt.ylim(0.0,1.0)
          ax.set_xlabel('false positive')
          ax.set_ylabel('true positive')
          plt.title(f'{algstr} trk: {tracknum}, {event_date}, {datetime.strftime(datetimes[i],'%y-%m-%d')}')
          fig.savefig(plotname,dpi=300,bbox_inches="tight")
          plotnum += 1
    plt.close(fig)

  return dist1ds,change,tp,fp,tn,fn

def accuracy(change,ref_aoi,Nconfirm,post_idxs):
  #Nref_tot = np.prod(change[0][0].shape)
  #Nref_tot = np.sum(np.sum(mask))
  Nref_tot = len(ref_aoi)
  Nref_pos = np.sum(np.sum(ref_aoi))
  Nref_neg = Nref_tot - Nref_pos
  # true positives
  frac_tp = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 1,ref_aoi == 1))))/Nref_pos
    for c in c1] for c1 in cposti]
    for cposti in change]
  # false positives
  frac_fp = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 1,ref_aoi == 0))))/Nref_neg
    for c in c1] for c1 in cposti]
    for cposti in change]
  # true negatives
  frac_tn = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 0,ref_aoi == 0))))/Nref_neg
    for c in c1] for c1 in cposti]
    for cposti in change]
  # false negatives
  frac_fn = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 0,ref_aoi == 1))))/Nref_pos
    for c in c1] for c1 in cposti]
    for cposti in change]
  return frac_tp,frac_fp,frac_tn,frac_fn

def accuracy2(change,refs,Nconfirm,post_idxs,mask):
  #Nref_tot = np.prod(change[0][0].shape)
  Nref_tot = np.sum(np.sum(mask))
  Nrefs_pos = [np.sum(np.sum(ref)) for ref in refs]
  Nrefs_neg = [Nref_tot - np.sum(np.sum(ref)) for ref in refs]
  Nrefs_pos = [N if N > 0 else 1 for N in Nrefs_pos]
  # true positives
  frac_tp = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 1,ref == 1,mask))))/Nref_pos
    for c in c1] for c1 in cposti]
    for cposti,ref,Nref_pos in zip(change,refs,Nrefs_pos)]
  # false positives
  frac_fp = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 1,ref == 0,mask))))/Nref_neg
    for c in c1] for c1 in cposti]
    for cposti,ref,Nref_neg in zip(change,refs,Nrefs_neg)]
  # true negatives
  frac_tn = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 0,ref == 0,mask))))/Nref_neg
    for c in c1] for c1 in cposti]
    for cposti,ref,Nref_neg in zip(change,refs,Nrefs_neg)]
  # false negatives
  frac_fn = [[[np.sum(np.sum(
    np.logical_and.reduce((c == 0,ref == 1,mask))))/Nref_pos
    for c in c1] for c1 in cposti]
    for cposti,ref,Nref_pos in zip(change,refs,Nrefs_pos)]
  return frac_tp,frac_fp,frac_tn,frac_fn

