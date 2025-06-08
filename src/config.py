# src/config.py

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study for MRT Inflow Forecasting with Darts RandomForest"
    )
    # 資料路徑與目標欄位
    parser.add_argument(
        "--raw-csv", type=str, required=True, help="原始資料 CSV 檔案路徑"
    )
    parser.add_argument(
        "--series-column",
        type=str,
        default="in",
        help="目標欄位名稱 (e.g. 'in' 或 'out')",
    )

    # 共變量欄位
    parser.add_argument(
        "--past-covariates",
        nargs="+",
        type=str,
        default=[],
        help="過去共變量欄位清單"
    )
    parser.add_argument(
        "--future-covariates",
        nargs="+",
        type=str,
        default=[],
        help="未來共變量欄位清單",
    )

    # 滯後與輸出長度
    parser.add_argument("--lags", type=int, default=24, help="target 過去滯後階數")
    parser.add_argument(
        "--lags-past-covariates",
        type=int,
        default=24,
        help="past covariates 過去滯後階數",
    )
    parser.add_argument(
        "--lags-future-covariates",
        nargs="+",         # 一定要加這行，否則只會讀到 int
        type=int,
        default=[0],
        help="future covariates 的 lag list"
    )
    parser.add_argument(
        "--output-chunk-length", type=int, default=1, help="一次預測步數"
    )

    # 切分比例與亂數種子
    parser.add_argument(
        "--train-test-split", type=float, default=0.7, help="訓練/測試切分比例"
    )
    parser.add_argument("--random-seed", type=int, default=42, help="全域亂數種子")

    # 結果輸出目錄
    parser.add_argument(
        "--results-dir", type=str, default="results", help="結果儲存根目錄"
    )

    # 預設模型參數（非 gridsearch）
    parser.add_argument(
        "--default-n-estimators",
        type=int,
        default=100,
        help="RandomForest n_estimators 預設值",
    )
    parser.add_argument(
        "--default-max-depth",
        type=int,
        default=15,
        help="RandomForest max_depth 預設值 (None 代表不限制)",
    )

    
    # Hyperparameter GridSearch
    parser.add_argument(
        "--do-gridsearch", action="store_true", help="是否執行 GridSearch"
    )
    parser.add_argument(
        "--param-n-estimators",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="GridSearch n_estimators 候選清單",
    )
    parser.add_argument(
        "--param-max-depth",
        nargs="+",
        type=int,
        default=[5, 10],
        help="GridSearch max_depth 候選清單",
    )
    parser.add_argument(
        "--param-min-samples-split",
        nargs="+",
        type=int,
        default=None,
        help="RandomForest min_samples_split",
    )
    parser.add_argument(
        "--param-min-samples-leaf",
        nargs="+",
        type=int,
        default=None,
        help="RandomForest min_samples_leaf",
    )
    parser.add_argument("--grid-start", type=int,
                        default=32, help="GridSearch start")
    parser.add_argument(
        "--grid-horizon", type=int, default=3, help="GridSearch forecast_horizon"
    )
    parser.add_argument("--grid-n-jobs", type=int,
                        default=-1, help="GridSearch n_jobs")
    parser.add_argument(
        "--grid-verbose", action="store_true", help="GridSearch verbose"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="若指定則將 log 輸出到檔案"
    )

    # Rolling forecast
    parser.add_argument(
        "--rolling-forecast",
        action="store_true",
        help="是否使用 rolling forecast (historical_forecasts)",
    )
    parser.add_argument(
        "--forecast-horizon", type=int, default=1, help="rolling forecast 每次預測步數"
    )
    parser.add_argument(
        "--retrain-per-step",
        action="store_true",
        help="rolling forecast 每步是否重新 fit 模型",
    )

    return parser.parse_args()
