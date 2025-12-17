import boto3
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
import rasterio
from tqdm import tqdm

from dist_s1_enumerator import enumerate_one_dist_s1_product


def get_burst_id(opera_rtc_id: str) -> str:
    return opera_rtc_id.split("_")[3]


def get_track_number(burst_id: str) -> int:
    return int(burst_id.split("-")[0][1:])


def get_acq_time(opera_rtc_id: str) -> pd.Timestamp:
    return pd.Timestamp(opera_rtc_id.split("_")[4])


def get_rtc_input_data(layer_path: str | Path) -> dict:
    with rasterio.open(layer_path) as ds:
        tags = ds.tags()

    rtc_inputs = {
        "post_rtc_opera_ids": tags["post_rtc_opera_ids"].split(","),
        "pre_rtc_opera_ids": tags["pre_rtc_opera_ids"].split(","),
        "mgrs_tile_id": tags["mgrs_tile_id"],
    }

    return rtc_inputs


def format_rtc_input_data(rtc_data: dict) -> pd.DataFrame:
    n_pre = len(rtc_data["pre_rtc_opera_ids"])
    df_pre = pd.DataFrame({"opera_id": rtc_data["pre_rtc_opera_ids"], "input_category": ["pre"] * n_pre})
    n_post = len(rtc_data["post_rtc_opera_ids"])
    df_post = pd.DataFrame({"opera_id": rtc_data["post_rtc_opera_ids"], "input_category": ["post"] * n_post})
    df = pd.concat([df_pre, df_post])
    df["opera_id_trunc"] = df.opera_id.map(lambda opera_id: "_".join(opera_id.split("_")[:5]))

    df["jpl_burst_id"] = df.opera_id.map(get_burst_id)
    df["track_number"] = df.jpl_burst_id.map(get_track_number)
    df["mgrs_tile_id"] = rtc_data["mgrs_tile_id"]
    df["acq_dt"] = df.opera_id.map(get_acq_time)
    return df


def download_s3_prefixes(s3_locations: list[str], out_dir: Path = Path("out/")) -> list[Path]:
    s3 = boto3.client("s3")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    objects_to_download = []
    for s3_location in s3_locations:
        parsed = urlparse(s3_location)
        assert parsed.scheme == "s3", f"Expected s3:// URI, got {s3_location}"
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        print(f"{bucket}/{prefix}")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                key_path = Path(key)

                if len(key_path.parts) >= 2:
                    dest_path = out_path / key_path.parts[-2] / key_path.name
                else:
                    dest_path = out_path / key_path.name

                objects_to_download.append((bucket, key, dest_path))

    dest_paths = []
    for bucket, key, dest_path in tqdm(objects_to_download, desc="Downloading"):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest_path))
        dest_paths.append(dest_path)

    return dest_paths


def validate_inputs_of_one(tif_path: str | Path, max_pre_imgs_per_burst: tuple[int, int, int] = (3, 3, 4)) -> bool:
    data = get_rtc_input_data(tif_path)
    df_prod = format_rtc_input_data(data)

    issues = []

    track_numbers = df_prod.track_number.unique().tolist()
    if len(track_numbers) > 1:
        if abs(track_numbers[0] - track_numbers[1]) > 1:
            issues.append(f"Too many track numbers present: {track_numbers}")

    post_ind = df_prod.input_category == "post"
    df_post = df_prod[post_ind].reset_index(drop=True)

    pre_ind = df_prod.input_category == "pre"
    df_pre = df_prod[pre_ind].reset_index(drop=True)

    post_time_delta = df_post.acq_dt.max() - df_post.acq_dt.min()
    if (post_time_delta).days > 1:
        issues.append(f"Post-dates span too long: {post_time_delta}")

    df_product_expected = enumerate_one_dist_s1_product(
        df_prod.mgrs_tile_id.iloc[0],
        track_number=track_numbers[0],
        post_date=str(df_post.acq_dt.min().date()),
        lookback_strategy="multi_window",
        delta_lookback_days=(1095, 730, 365),
        max_pre_imgs_per_burst=max_pre_imgs_per_burst,
    )
    df_product_expected["opera_id_trunc"] = df_product_expected.opera_id.map(
        lambda opera_id: "_".join(opera_id.split("_")[:5])
    )

    post_ind = df_product_expected.input_category == "post"
    df_product_expected_post = df_product_expected[post_ind].reset_index(drop=True)

    pre_ind = df_product_expected.input_category == "pre"
    df_product_expected_pre = df_product_expected[pre_ind].reset_index(drop=True)

    burst_id_expected = sorted(df_product_expected.jpl_burst_id.unique().tolist())
    bust_id_prod = sorted(df_prod.jpl_burst_id.unique().tolist())

    if not (burst_id_expected == bust_id_prod):
        issues.append(f"Burst ID mismatch - expected: {burst_id_expected}, found: {bust_id_prod}")

    pre_rtc_ids_expected_but_not_found = [
        rtc_id
        for rtc_id in df_product_expected_pre.opera_id_trunc.tolist()
        if rtc_id not in df_pre.opera_id_trunc.tolist()
    ]
    if pre_rtc_ids_expected_but_not_found:
        issues.append(f"Pre RTC IDs expected but not found: {pre_rtc_ids_expected_but_not_found}")

    pre_found_but_not_expected = [
        rtc_id
        for rtc_id in df_pre.opera_id_trunc.tolist()
        if rtc_id not in df_product_expected_pre.opera_id_trunc.tolist()
    ]
    if pre_found_but_not_expected:
        issues.append(f"Pre RTC IDs found but not expected: {pre_found_but_not_expected}")

    post_rtc_ids_expected_but_not_found = [
        rtc_id
        for rtc_id in df_product_expected_post.opera_id_trunc.tolist()
        if rtc_id not in df_post.opera_id_trunc.tolist()
    ]
    if post_rtc_ids_expected_but_not_found:
        issues.append(f"Post RTC IDs expected but not found: {post_rtc_ids_expected_but_not_found}")

    post_found_but_not_expected = [
        rtc_id
        for rtc_id in df_post.opera_id_trunc.tolist()
        if rtc_id not in df_product_expected_post.opera_id_trunc.tolist()
    ]
    if post_found_but_not_expected:
        issues.append(f"Post RTC IDs found but not expected: {post_found_but_not_expected}")

    if issues:
        raise ValueError("\n".join(issues))
