# src/experiments.py

import logging
import os
import json
from darts import TimeSeries
from src import data_loader, model as model_module, utils

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, cfg, name: str, past_covariate_cols: list):
        self.cfg = cfg
        self.name = name
        self.past_covariate_cols = past_covariate_cols
        self.metrics = {}

    def run(self):
        logger.info(f"[{self.name}] Start experiment")
        print(f"self.cfg.rolling_forecast: {self.cfg.rolling_forecast}")
        # 1. 載入 & 前處理
        df = data_loader.load_raw_data(self.cfg.raw_csv)
        df = data_loader.preprocess_df(df)
        target_ts, past_cov_ts, future_cov_ts = data_loader.build_timeseries(
            df,
            series_col=self.cfg.series_column,
            past_covariate_cols=self.past_covariate_cols,
            future_covariate_cols=self.cfg.future_covariates,
        )
        if past_cov_ts is not None:
            logger.info(
                f"[{self.name}] Using past covariates: {self.past_covariate_cols}"
            )
        else:
            logger.info(f"[{self.name}] No past covariates used")
        # 2. 切分 train / test
        train_target, val_target, test_target = data_loader.train_val_test_split_series(
            target_ts, train_ratio=0.70, val_ratio=0.15
        )

        train_future_cov, val_future_cov, test_future_cov = (
            data_loader.train_val_test_split_series(
                future_cov_ts, train_ratio=0.70, val_ratio=0.15
            )
        )
        train_past_cov, val_past_cov, test_past_cov = (None, None, None)
        if past_cov_ts:
            train_past_cov, val_past_cov, test_past_cov = (
                data_loader.train_val_test_split_series(
                    past_cov_ts, train_ratio=0.70, val_ratio=0.15
                )
            )

        print(self.cfg)
        # 3. 建立模型或 GridSearch
        if self.cfg.do_gridsearch:
            logger.info(f"[{self.name}] Running GridSearch")
            model, best_params = model_module.run_gridsearch(
                cfg=self.cfg,
                train_series=train_target,
                train_past_cov=train_past_cov,
                train_future_cov=train_future_cov,
                val_series=val_target,
                val_past_cov=val_past_cov,
                val_future_cov=val_future_cov,
                use_past_covariates=bool(train_past_cov),
            )
            self.metrics["best_params"] = json.dumps(best_params)
        else:
            logger.info(f"[{self.name}] Creating default model")
            model = model_module.create_model(
                self.cfg,
                lags=self.cfg.lags,
                lags_past_cov=self.cfg.lags_past_covariates,
                lags_future_cov=self.cfg.lags_future_covariates,
                output_chunk_length=self.cfg.output_chunk_length,
                use_past_covariates=bool(train_past_cov),
            )

        # 4. Fit
        logger.info(f"[{self.name}] Fitting model")
        model.fit(
            series=train_target.concatenate(val_target),
            past_covariates=(train_past_cov.concatenate(val_past_cov) if bool(train_past_cov) else None),
            future_covariates=train_future_cov.concatenate(val_future_cov),
        )

        # 5. Predict
        logger.info(f"[{self.name}] Making predictions")
        print(f"self.cfg.rolling_forecast:, {self.cfg.rolling_forecast}")
        # 5. Predict
        logger.info(f"[{self.name}] Making predictions")

        if self.cfg.rolling_forecast:
            logger.info(f"[{self.name}] Rolling forecast")

            # 1) 用 historical_forecasts 一次取得所有 rolling-window 預測（每次預測 h=3 步）
            forecasts = model.historical_forecasts(
                series=target_ts,
                past_covariates=past_cov_ts,
                future_covariates=future_cov_ts,
                start=0.85,
                forecast_horizon=1,
                stride=1,
                retrain=False,
                last_points_only=True,
                verbose=True,
                # enable_optimization=False,
            )

            logger.info(f"[{self.name}] Direct predict")
            preds_ts = model.predict(
                n=len(test_target),
                past_covariates=past_cov_ts,
                future_covariates=test_future_cov,
            )
            # 保留你原本的單次 predict 評估
            m = utils.compute_metrics(test_target, preds_ts)
            self.metrics.update(m)
            logger.info(f"[{self.name}] Metrics: MAE={m['MAE']:.4f}, MSE={m['MSE']:.4f}")

        # 7. 儲存結果
        from pathlib import Path
        # 確保 results_dir 存在
        results_dir = Path(self.cfg.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 將 preds_ts / test_target 轉成 pandas.Series，並呼叫 utils
        preds_series  = preds_ts.pd_series().rename("prediction")
        actuals_series= test_target.pd_series().rename("actual")

        utils.save_predictions(
            self.name,
            preds_series,
            results_dir,
            actuals=actuals_series,
        )
        utils.append_metrics_summary(
            self.name,
            self.metrics,
            results_dir,
        )

        logger.info(f"[{self.name}] Saved predictions and metrics to {results_dir}")

        logger.info(f"[{self.name}] Experiment finished")
