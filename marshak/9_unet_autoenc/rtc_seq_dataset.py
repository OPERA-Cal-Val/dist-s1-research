import bisect
import concurrent.futures
from functools import lru_cache
from pathlib import Path

import backoff
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
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


@lru_cache
def open_patch_table() -> pd.DataFrame:
    return pd.read_json("../6_torch_dataset/dist_s1_patch_lut.json.zip")


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
        patch_table: str = None,
        n_pre_imgs=4,
        root=Path("opera_rtc_data"),
        download=False,
        transform=None,
    ):
        self.download = download
        self.root = root

        self.rtc_table = rtc_table if rtc_table is not None else open_rtc_table()
        self.patch_table = (
            patch_table if patch_table is not None else open_patch_table()
        )
        self.n_pre_imgs = n_pre_imgs

        self.burst_ids = list(self.rtc_table.jpl_burst_id.unique())
        self._total_samples = None
        self._df_count = None

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

    @property
    def df_count(self) -> pd.DataFrame:
        """df_count is a dataframe each row is keyed by burst_id and which also contains
        running samples which is a cumsum to understand a runny tally of samples for __getitem__.
        """
        if self._df_count is None:
            df_acq_per_burst = (
                self.rtc_table.groupby("jpl_burst_id")
                .size()
                .reset_index(name="acq_per_burst")
            )
            df_patch_per_burst = (
                self.patch_table.groupby("jpl_burst_id")
                .size()
                .reset_index(name="patch_per_burst")
            )

            df_count = pd.merge(
                df_acq_per_burst, df_patch_per_burst, on="jpl_burst_id"
            ).reset_index(drop=True)
            # subtract n preimages and 1 post images
            df_count["total_samples_per_burst"] = (
                np.maximum(df_count["acq_per_burst"].values - self.n_pre_imgs - 1, 0)
                * df_count["patch_per_burst"]
            )
            df_count["running_samples"] = df_count["total_samples_per_burst"].cumsum()
            self._df_count = df_count
            return self._df_count
        else:
            return self._df_count

    @property
    def total_samples(self) -> int:
        if self._total_samples is None:
            df_count = self.df_count
            self._total_samples = df_count["total_samples_per_burst"].sum()
            return self._total_samples
        else:
            return self._total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # *first* burst_id whose running samples value is larger than idx.
        # note the df_count.iloc[burst_idx].running_samples >= idx
        burst_idx = bisect.bisect_left(self.df_count.running_samples.tolist(), idx)
        assert self.df_count.iloc[burst_idx].running_samples >= idx

        burst_id = self.df_count.iloc[burst_idx].jpl_burst_id
        # As we did for df_count, we need to subtract the n preimages and 1 pre-image
        acq_for_burst_lookup = (
            self.df_count.iloc[burst_idx].acq_per_burst - self.n_pre_imgs - 1
        )
        patches_for_burst = self.df_count.iloc[burst_idx].patch_per_burst

        # Key here is we need the n samples < idx to determine how to sample patch and acq time
        total_samples_running_idx = (
            self.df_count.iloc[burst_idx - 1].running_samples if burst_idx > 0 else 0
        )
        assert total_samples_running_idx <= idx

        acq_idx = (idx - total_samples_running_idx) % acq_for_burst_lookup
        df_ts_t = self.rtc_table[self.rtc_table.jpl_burst_id == burst_id].reset_index(
            drop=True
        )
        # Add 1 to index to get post image
        # df_ts is a dataframe of n_preimgs + 1 in dim 0 to provide metadata for one sample of sequence
        df_ts = df_ts_t.iloc[acq_idx : acq_idx + self.n_pre_imgs + 1].reset_index(
            drop=True
        )
        assert df_ts.shape[0] == (
            self.n_pre_imgs + 1
        ), f"Issue with {idx=}, {burst_idx=}"

        patch_idx = (idx - total_samples_running_idx) % patches_for_burst
        patch_burst = self.patch_table[
            self.patch_table.jpl_burst_id == burst_id
        ].reset_index(drop=True)
        patch_data = patch_burst.iloc[patch_idx].to_dict()

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
