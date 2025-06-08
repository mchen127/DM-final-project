# src/data_loader.py

import os
import pandas as pd
from darts import TimeSeries


def load_raw_data(path: str) -> pd.DataFrame:
    """讀取原始 CSV 到 DataFrame"""
    return pd.read_csv(path)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """合併年月日時，設定 datetime index，並移除原始欄位"""
    # 1. 合併 year, month, date, time → datetime
    df["datetime"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-"
        + df["date"].astype(str)
        + " "
        + df["time"].astype(str)
        + ":00:00"
    )
    df = df.set_index("datetime").sort_index()
    # 移除拆分出 datetime 的欄位
    df = df.drop(columns=["year", "month", "date", "time"])
    # 若需對 weekday, is_holiday 做 one-hot，可在這裡擴充
    # df = pd.get_dummies(df, columns=["weekday", "is_holiday"], prefix=["wd", "hol"])
    
    # 2. 把 -9999.0 當成 NaN
    df = df.replace(-9999.0, pd.NA)

    # 3. 針對 target 欄位（in / out）強制 dropna
    #    這裡假設後面才會指定 series_col
    #    你也可以在 build_timeseries 之前加
    # df = df.dropna(subset=[series_col])

    # 4. 其餘共變量：前向填補，再後向填補
    df = df.fillna(method="ffill").fillna(method="bfill")
    print(f"Preprocessed DataFrame shape: {df.shape}")
    print(f"Columns after preprocessing: {df.columns.tolist()}")
    return df


def build_timeseries(
    df: pd.DataFrame, series_col: str, past_covariate_cols: list, future_covariate_cols: list = None
) -> tuple[TimeSeries, TimeSeries]:
    """
    建立 Darts TimeSeries:
      - target_ts: 單一欄位
      - covariates_ts: 多欄位或 None
    """
    # 確保 target 無缺失
    df = df.dropna(subset=[series_col])

    target_ts = TimeSeries.from_series(df[series_col])
    future_cov_ts = TimeSeries.from_dataframe(df[future_covariate_cols])
    past_cov_ts = None
    if past_covariate_cols:
        print(f"Building covariates TimeSeries with columns: {past_covariate_cols}")
        past_cov_ts = TimeSeries.from_dataframe(df[past_covariate_cols])
    return target_ts, past_cov_ts, future_cov_ts


def train_val_test_split_series(
    ts: TimeSeries, train_ratio: float, val_ratio: float
) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    """
    把 ts 切成 train/val/test 三段。
      - train_ratio + val_ratio <= 1
      - test_ratio = 1 - train_ratio - val_ratio
    """
    total = len(ts)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_ts = ts[:train_end]
    val_ts = ts[train_end:val_end]
    test_ts = ts[val_end:]
    return train_ts, val_ts, test_ts


def train_test_split_series(
    ts: TimeSeries, split_ratio: float
) -> tuple[TimeSeries, TimeSeries]:
    """依比例在時間軸上切成 train / test"""
    total_pts = len(ts)
    train_size = int(total_pts * split_ratio)
    train_ts = ts[:train_size]
    test_ts = ts[train_size:]
    return train_ts, test_ts
