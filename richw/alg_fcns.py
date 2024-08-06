#! /usr/bin/env python

import numpy as np
from distmetrics import compute_mahalonobis_dist_1d
from tqdm import tqdm

# Functions to support disturbance algorithms

def window_mahalanobis_1d_running(window_data: list[list], n_pre_img: int = 5):
    arrs = [np.array(window_l).reshape((3, 3)) for window_l in window_data]
    N = len(arrs)
    dists = [compute_mahalonobis_dist_1d(arrs[k- n_pre_img : k ], arrs[k]) if k > n_pre_img else None for k in tqdm(range(N))]
    return dists

def window_mahalanobis_1d_workflow(window_data: list[list], T: int = 3, n_pre_img: int = 5, n_post_imgs_to_confirm: int = 3):
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


