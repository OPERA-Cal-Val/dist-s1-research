import pandas as pd


def get_last_index_before(timestamps: pd.DatetimeIndex, target: pd.Timestamp) -> int | None:
    timestamps_series = pd.Series(timestamps)
    mask = timestamps_series < target
    if not mask.any():
        return None
    return mask.idxmin() - 1
