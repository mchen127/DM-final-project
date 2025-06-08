# src/utils.py

import os
import json
import logging
import pandas as pd
from darts.metrics import mae, mse


def setup_logging(level: str, log_file: str = None):
    """設定全域 logging"""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def compute_metrics(true_ts, pred_ts) -> dict:
    """計算 MAE、MSE"""
    return {
        "MAE": mae(true_ts, pred_ts),
        "MSE": mse(true_ts, pred_ts),
    }


def save_predictions(
    exp_name: str, preds: pd.Series, results_dir: str, actuals: pd.Series = None
):
    """將預測結果輸出為 CSV"""
    out_dir = os.path.join(results_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    df = preds.reset_index()
    df.columns = ["datetime", "prediction"]

    if actuals is not None:
        df["actual"] = actuals.reset_index(drop=True)

    df.to_csv(os.path.join(out_dir, f"{exp_name}.csv"), index=False)


def append_metrics_summary(exp_name: str, metrics: dict, results_dir: str):
    """將 metrics 加入 summary CSV"""
    out_dir = os.path.join(results_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    row = {"experiment": exp_name, **metrics}
    # 如果不存在則寫入 header
    if not os.path.exists(summary_path):
        df = pd.DataFrame([row])
        df.to_csv(summary_path, index=False)
    else:
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(summary_path, index=False)
