# src/model.py

from darts.models.forecasting.random_forest import RandomForest
from darts import TimeSeries
from darts.metrics import mae

def create_model(cfg, lags, lags_past_cov, lags_future_cov, output_chunk_length, use_past_covariates: bool):
    """建立不經 GridSearch 的 RandomForest 實例"""
    # 如果沒共變量，就把 lag covariates 參數設為 None
    lpc = lags_past_cov if use_past_covariates else None

    return RandomForest(
        lags=lags,
        lags_past_covariates=lpc,
        lags_future_covariates=lags_future_cov,
        output_chunk_length=output_chunk_length,
        n_estimators=cfg.default_n_estimators,
        max_depth=cfg.default_max_depth,
        random_state=cfg.random_seed,
    )


def run_gridsearch(
    cfg,
    train_series,
    train_past_cov,
    train_future_cov,
    val_series,
    val_past_cov,
    val_future_cov,
    use_past_covariates: bool,
):
    """對 RandomForest 進行 hyperparameter grid search"""
    lpc = cfg.lags_past_covariates if use_past_covariates else None
    parameters = {
        "lags": [cfg.lags],
        "lags_past_covariates": [lpc],
        "lags_future_covariates": [cfg.lags_future_covariates],
        "output_chunk_length": [cfg.output_chunk_length],
        "n_estimators": cfg.param_n_estimators,
        "max_depth": cfg.param_max_depth,
        "min_samples_split": cfg.param_min_samples_split,
        "min_samples_leaf": cfg.param_min_samples_leaf,
        "random_state": [cfg.random_seed],
    }
    print(f"Running gridsearch with parameters: {parameters}")
    best_model, best_params, _ = RandomForest.gridsearch(
        parameters=parameters,
        series=train_series,
        past_covariates=(
            train_past_cov.concatenate(val_past_cov) if train_past_cov else None
        ),
        future_covariates=train_future_cov.concatenate(val_future_cov),
        # start=cfg.grid_start,
        # forecast_horizon=cfg.grid_horizon,
        val_series=val_series,  # <<< 使用驗證集
        # val_past_covariates=val_past_cov,  # <<< 如有
        # val_future_covariates=val_future_cov,  # <<< 如有
        metric=mae,
        n_jobs=cfg.grid_n_jobs,
        verbose=cfg.grid_verbose,
        # enable_optimization=False,
    )

    print(f"Best parameters found: {best_params}")
    print(f"Best model: {best_model}")
    return best_model, best_params
