import bisect
import concurrent.futures
import random
from functools import lru_cache
from pathlib import Path

import backoff
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import geopandas as gpd
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from requests.exceptions import HTTPError
from skimage.restoration import denoise_tv_bregman
from torch.utils.data import Dataset
from tqdm import tqdm


def despeckle_one(X: np.ndarray, reg_param=5, noise_floor_db=-22, preserve_nans=False) -> np.ndarray:
    X_c = np.clip(X, 1e-7, 1)
    X_db = 10 * np.log10(X_c, out=np.full(X_c.shape, np.nan), where=(~np.isnan(X_c)))
    X_db[np.isnan(X_c)] = noise_floor_db
    X_db_dspkl = denoise_tv_bregman(
        X_db, weight=1.0 / reg_param, isotropic=True, eps=1e-3
    )
    X_dspkl = np.power(10, X_db_dspkl / 10.0)
    if preserve_nans:
        X_dspkl[np.isnan(X)] = np.nan
    X_dspkl = np.clip(X_dspkl, 0, 1)
    return X_dspkl


@lru_cache
def open_rtc_table() -> pd.DataFrame:
    return pd.read_json("../4_rtc_organization/rtc_s1_table.json.zip")


def read_window(url: str, x_start, y_start, x_stop, y_stop) -> np.ndarray:
    rows = (y_start, y_stop)
    cols = (x_start, x_stop)
    window = Window.from_slices(rows=rows, cols=cols)

    with rasterio.open(url) as ds:
        X = ds.read(1, window=window).astype(np.float32)
        t_window = ds.window_transform(window)
        crs = ds.crs

    p = DefaultGTiffProfile()
    p["dtype"] = "float32"
    p["transform"] = t_window
    p["crs"] = crs
    p["nodata"] = np.nan
    p["count"] = 1
    p["height"], p["width"] = X.shape

    return X, p


@backoff.on_exception(
    backoff.expo,
    [ConnectionError, HTTPError],
    max_tries=10,
    max_time=60,
    jitter=backoff.full_jitter,
)
def localize_one_rtc(url: str | Path, ts_dir: str | Path = Path(".")) -> Path:
    local_fn = url.split("/")[-1]
    burst_id = local_fn.split("_")[3]
    out_dir = Path(ts_dir) / burst_id
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / local_fn

    if out_path.exists():
        return out_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return out_path


class SeqDistDataset(Dataset):
    def __init__(
        self,
        rtc_table: str = None,
        patch_data_dir: str = None,
        n_pre_imgs=4,
        root=Path("opera_rtc_data"),
        download=False,
    ):
        self.download = download
        self.root = root

        self.rtc_table = rtc_table if rtc_table is not None else open_rtc_table()
        self.patch_data_dir = patch_data_dir or Path("../6_torch_dataset/burst_patch_data")

        self.n_pre_imgs = n_pre_imgs

        self.burst_ids = list(self.rtc_table.jpl_burst_id.unique())

        if self.download:
            self.download_rtc_data()

    def download_rtc_data(self, max_workers=50):
        n = self.rtc_table.shape[0]

        def localize_one_rtc_p(url):
            return localize_one_rtc(url, ts_dir=self.root)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            vv_loc_paths = list(
                tqdm(
                    executor.map(localize_one_rtc_p, self.rtc_table.rtc_s1_vv_url),
                    total=n,
                    desc="downloading vv",
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            vh_loc_paths = list(
                tqdm(
                    executor.map(localize_one_rtc_p, self.rtc_table.rtc_s1_vh_url),
                    total=n,
                    desc="downloading vh",
                )
            )

        self.rtc_table["rtc_s1_vv_loc_path"] = [str(p) for p in vv_loc_paths]
        self.rtc_table["rtc_s1_vh_loc_path"] = [str(p) for p in vh_loc_paths]

    def __len__(self):
        return len(self.burst_ids)

    def __getitem__(self, idx):

        burst_id = self.burst_ids[idx]
        while True:
            df_ts_full = self.rtc_table[self.rtc_table.jpl_burst_id == burst_id].reset_index(drop=True)
            n_acqs_for_burst = df_ts_full.shape[0]
            # 20 is arbitrary - but don't want time series with less than 20 acquisitions over 2 years time.
            if n_acqs_for_burst < max(20, self.n_pre_imgs + 1):
                continue
            else:
                N = df_ts_full.shape[0] - self.n_pre_imgs - 1
                i = random.randint(0, N)
                df_ts = df_ts_full.iloc[i: i + self.n_pre_imgs + 1].reset_index(drop=True)
                break

        df_burst_patches = gpd.read_file(self.patch_data_dir / f'{burst_id}.geojson')
        M = df_burst_patches.shape[0]
        j = random.randint(0, M)
        patch_data = df_burst_patches.iloc[j].to_dict()

        def read_window_p(url: str):
            return read_window(
                url,
                patch_data["x_start"],
                patch_data["y_start"],
                patch_data["x_stop"],
                patch_data["y_stop"],
            )

        vv_loc = df_ts.rtc_s1_vv_url if not self.download else df_ts.rtc_s1_vv_loc_path
        vh_loc = df_ts.rtc_s1_vh_url if not self.download else df_ts.rtc_s1_vh_loc_path

        vv_data, ps = zip(
            *[read_window_p(url) for url in tqdm(vv_loc, desc="loading vv")]
        )
        vh_data, _ = zip(
            *[read_window_p(url) for url in tqdm(vh_loc, desc="loading vh")]
        )

        nodata_masks = [np.isnan(vv) for vv in vv_data]
        nodata_masks_stack = np.stack(nodata_masks, axis=0).astype(bool)

        vv_data_d = [despeckle_one(vv, preserve_nans=False) for vv in tqdm(vv_data, desc="tv for vv")]
        vh_data_d= [despeckle_one(vh, preserve_nans=False) for vh in tqdm(vh_data, desc="tv for vh")]

        # A list of 2 x H X W imagery
        acq_data = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(vv_data_d, vh_data_d)]

        # Input for modeling
        # pre img is n_pre_imgs X 2 X H X W
        pre_imgs = np.stack(acq_data[: self.n_pre_imgs], axis=0)
        # post img is 2 X H X W
        post_img = acq_data[-1]

        # additional metadata
        pre_dts = df_ts.iloc[: self.n_pre_imgs].acq_datetime.tolist()
        post_dt = df_ts.loc[self.n_pre_imgs, "acq_datetime"]

        return {
            "pre_imgs": torch.from_numpy(pre_imgs),
            "post_img": torch.from_numpy(post_img),
            "nodata_masks": torch.from_numpy(nodata_masks_stack),
            "pre_dts": pre_dts,
            "post_dt": post_dt,
            "profile": ps[0],
        }
