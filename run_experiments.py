#!/usr/bin/env python3
# run_experiments.py

import logging
from tqdm import tqdm

from src.config import parse_args
from src.utils import setup_logging
from src.experiments import Experiment


def main():
    # 1. 解析參數並設定 logging
    cfg = parse_args()
    setup_logging(cfg.log_level, cfg.log_file)
    logger = logging.getLogger("run_experiments")
    logger.info("Starting Ablation experiments")

    # 2. 定義 Ablation Study 之實驗組合
    experiments = []
    target = "out"
    # (1) 不使用任何共變量
    experiments.append({"name": f"{target}_no_weather", "covariates": [], })
    # (2) 使用全部 past_covariates
    experiments.append({"name": f"{target}_full_weather", "covariates": cfg.past_covariates})
    # (3) 拆掉每個欄位
    for col in cfg.past_covariates:
        reduced = [c for c in cfg.past_covariates if c != col]
        experiments.append({"name": f"{target}_drop_{col}", "covariates": reduced})

    # 3. 迴圈執行所有實驗，並顯示進度條
    for exp_cfg in tqdm(experiments, desc="Running experiments", ncols=80):
        name = exp_cfg["name"]
        covs = exp_cfg["covariates"]

        logger.info(f"=== Begin experiment: {name} ===")
        exp = Experiment(cfg, name, covs)
        exp.run()
        logger.info(f"=== Finished experiment: {name} ===\n")

    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    main()
