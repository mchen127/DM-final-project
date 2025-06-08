python run_experiments.py \
  --raw-csv data/raw/data.csv \
  --series-column in \
  --lags 72 \
  --lags-past-covariates 24 \
  --lags-future-covariates 0 \
  --output-chunk-length 3 \
  --past-covariates station_pressure sea_level_pressure	temperature	\
                    dew_point_temperature relative_humidity	vapor_pressure \
                    avg_wind_speed avg_wind_direction precipitation	\
                    precipitation_hours sunshine_hours pressure_trend \
                    solar_radiation	saturation_vapor_pressure uv_index \
  --future-covariates weekday is_holiday \
  --train-test-split 0.85 \
  --random-seed 42 \
  --results-dir results \
  \
  --do-gridsearch \
  --param-n-estimators 100 \
  --param-max-depth 15 \
  --param-min-samples-split 2 \
  --param-min-samples-leaf 1 \
  --grid-verbose \
  --grid-n-jobs -1 \
  \
  --rolling-forecast \
  \
  --log-level INFO \
  --log-file logs/exp.log
