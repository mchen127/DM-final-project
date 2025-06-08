## Project structure
```
project_root/
├── src/
│   ├── config.py           # 命令列參數定義
│   ├── data_loader.py      # 資料載入與前處理
│   ├── model.py            # RandomForest 建立、GridSearch
│   ├── experiments.py      # Experiment 類別：一次實驗完整流程
│   ├── utils.py            # Logging 設定、metrics、CSV 輸出
│   └── __init__.py
├── data/
│   ├── raw/                # 原始 CSV 檔
│   └── processed/          # 中繼檔（TimeSeries pickle）
├── results/
│   ├── predictions/        # 各實驗預測結果 CSV
│   └── metrics/            # metrics summary CSV
├── notebooks/
│   └── EDA.ipynb           # 探索性資料分析
├── tests/
│   ├── test_data_loader.py
│   └── test_model.py
├── requirements.txt
└── run_experiments.py      # CLI 入口：啟動所有 Ablation 實驗
```